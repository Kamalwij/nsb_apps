import datetime as dt
from typing import List, Optional, Tuple
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Helpers
# =========================

def pmt(rate: float, nper: int, pv: float) -> float:
    """
    Payment per period (Excel-like PMT).
    rate is per-period rate (monthly), nper is number of periods, pv is present value.
    Returns a negative value for the payment (cash outflow).
    """
    if nper <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return -(pv / nper)
    return -(pv * rate) / (1.0 - (1.0 + rate) ** (-nper))


def amortize_fixed(
    principal: float,
    annual_rate: float,
    months: int,
    start_date: Optional[dt.date] = None
) -> pd.DataFrame:
    """Standard fixed-rate amortization with constant EMI for the entire term."""
    rate_m = annual_rate / 12.0 / 100.0
    emi = -pmt(rate_m, months, principal)
    bal = principal
    rows = []
    d = start_date or dt.date.today().replace(day=1)

    for i in range(1, months + 1):
        interest = bal * rate_m
        principal_comp = emi - interest
        bal = max(0.0, bal - principal_comp)
        rows.append({
            "Period": i,
            "Date": d + pd.DateOffset(months=i-1),
            "Payment": emi,
            "Interest": interest,
            "Principal": principal_comp,
            "Balance": bal,
        })
        if bal <= 1e-6:
            break

    return pd.DataFrame(rows)


def amortize_variable(
    principal: float,
    periods: List[Tuple[int, float]],  # [(months, annual_rate%), ...]
    start_date: Optional[dt.date] = None
) -> pd.DataFrame:
    """
    Variable-rate schedule.
    At the start of each segment, payment is recalculated to amortize the remaining balance
    over the remaining months (common bank practice).
    """
    total_months = sum(m for m, _ in periods)
    d = start_date or dt.date.today().replace(day=1)
    rows = []
    bal = principal
    month_index = 1
    months_remaining = total_months

    for seg_idx, (seg_months, annual_rate) in enumerate(periods, start=1):
        if months_remaining <= 0 or bal <= 1e-6:
            break

        rate_m = annual_rate / 12.0 / 100.0
        emi = -pmt(rate_m, months_remaining, bal)  # re-amortize over remaining term

        for _ in range(seg_months):
            if months_remaining <= 0 or bal <= 1e-6:
                break
            interest = bal * rate_m
            principal_comp = emi - interest
            bal = max(0.0, bal - principal_comp)

            rows.append({
                "Period": month_index,
                "Date": d + pd.DateOffset(months=month_index-1),
                "Segment": seg_idx,
                "Rate (annual %)": annual_rate,
                "Payment": emi,
                "Interest": interest,
                "Principal": principal_comp,
                "Balance": bal,
            })

            month_index += 1
            months_remaining -= 1

    return pd.DataFrame(rows)


def amortize_hybrid_teaser_then_fixed(
    principal: float,
    teaser_months: int,
    teaser_rate_annual: float,
    remaining_months_rate_annual: float,
    total_months: int,
    start_date: Optional[dt.date] = None
) -> pd.DataFrame:
    """Hybrid = two-segment variable schedule (teaser then fixed)."""
    d = start_date or dt.date.today().replace(day=1)
    periods = [
        (teaser_months, teaser_rate_annual),
        (max(0, total_months - teaser_months), remaining_months_rate_annual),
    ]
    return amortize_variable(principal, periods, start_date=d)


def combine_tranche_schedules(schedules: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Sum multiple tranche schedules period-by-period.
    Assumes all tranches share the same Period/Date index and tenure.
    """
    if not schedules:
        return pd.DataFrame()

    norm = []
    for i, df in enumerate(schedules, start=1):
        d = df.copy()
        if "Rate (annual %)" not in d.columns:
            d["Rate (annual %)"] = np.nan
        if "Segment" not in d.columns:
            d["Segment"] = np.nan
        d = d[["Period", "Date", "Payment", "Interest", "Principal", "Balance"]]
        d = d.rename(columns={
            "Payment": f"Payment_{i}",
            "Interest": f"Interest_{i}",
            "Principal": f"Principal_{i}",
            "Balance": f"Balance_{i}",
        })
        norm.append(d)

    out = norm[0]
    for d in norm[1:]:
        out = out.merge(d, on=["Period", "Date"], how="outer").sort_values("Period")

    pay_cols = [c for c in out.columns if c.startswith("Payment_")]
    int_cols = [c for c in out.columns if c.startswith("Interest_")]
    pri_cols = [c for c in out.columns if c.startswith("Principal_")]
    bal_cols = [c for c in out.columns if c.startswith("Balance_")]

    out["Payment"] = out[pay_cols].sum(axis=1)
    out["Interest"] = out[int_cols].sum(axis=1)
    out["Principal"] = out[pri_cols].sum(axis=1)
    out["Balance"] = out[bal_cols].sum(axis=1)

    return out[["Period", "Date", "Payment", "Interest", "Principal", "Balance"]]


def solve_effective_rate_for_emi(total_principal: float, months: int, total_emi: float) -> float:
    """
    Find a single annual rate (%) such that PMT(rate_m, months, total_principal) == -total_emi.
    Bisection on [0%, 60%].
    """
    if months <= 0 or total_principal <= 0 or total_emi <= 0:
        return 0.0

    target = -total_emi  # PMT returns negative
    lo, hi = 0.0, 0.60   # annual
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pm = pmt(mid / 12.0, months, total_principal)  # negative
        # If pm is MORE negative than target, rate too high -> decrease hi
        if pm < target:
            hi = mid
        else:
            lo = mid
    return 100.0 * 0.5 * (lo + hi)  # annual %


def solve_effective_rate_from_schedule(principal: float, payments: list) -> float:
    """
    Effective annual rate from the FULL schedule (EMI can vary).
    Solve r so PV(payments discounted monthly at r) == principal.
    """
    if principal <= 0 or not payments:
        return 0.0

    def pv_of_payments(annual_r: float) -> float:
        rm = annual_r / 12.0
        pv = 0.0
        for t, pay in enumerate(payments, start=1):
            pv += pay / ((1.0 + rm) ** t)
        return pv

    lo, hi = 0.0, 0.60  # annual
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        pv_mid = pv_of_payments(mid)
        if pv_mid > principal:  # discounting too weak -> rate too low
            lo = mid
        else:
            hi = mid
    return 100.0 * 0.5 * (lo + hi)


def build_slabs_tranches(loan_amt: float, slabs_df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Given slabs: columns cap_lkr (cumulative cap) and annual_rate (%).
    Returns [(tranche_amount, annual_rate)] by filling slabs in order.
    """
    tranches = []
    remaining = loan_amt
    prev_cap = 0.0

    for row in slabs_df.itertuples(index=False):
        cap = float(row.cap_lkr) if pd.notna(row.cap_lkr) else float("inf")
        rate = float(row.annual_rate)
        band_capacity = max(0.0, cap - prev_cap)
        alloc = min(remaining, band_capacity)
        if alloc > 0:
            tranches.append((alloc, rate))
            remaining -= alloc
        prev_cap = cap
        if remaining <= 1e-6:
            break

    return tranches


# =========================
# UI
# =========================

st.set_page_config(page_title="NSB Loan Calculator", layout="wide")
st.title("NSB Loan Calculator")
st.caption("Fixed, Variable, Hybrid, Tiered-by-Amount (Slabs), Multi‑Tranche, and Tiered Slabs with per‑slab teaser")

with st.sidebar:
    st.header("Inputs")
    loan_amt = st.number_input("Loan Amount (LKR)", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%.2f")
    tenure_years = st.number_input("Tenure (years)", min_value=0.5, value=10.0, step=0.5, format="%.1f")
    months = int(round(tenure_years * 12))
    start_date = st.date_input("Start date", value=dt.date.today())

    mode = st.selectbox(
        "Mode",
        [
            "Fixed",
            "Variable (rate schedule)",
            "Hybrid (teaser then fixed)",
            "Tiered by amount (slabs)",
            "Multi‑tranche (principal slabs with own schedules)",
            "Tiered slabs (per‑slab teaser)",
        ]
    )

# =========================
# Compute schedule by mode
# =========================

if mode == "Fixed":
    fixed_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=9.50, step=0.05, format="%.2f")
    sched = amortize_fixed(loan_amt, fixed_rate, months, start_date=start_date)
    meta = {"mode": "Fixed", "effective_rate": fixed_rate, "emi": float(-pmt(fixed_rate/12/100, months, loan_amt))}

elif mode == "Variable (rate schedule)":
    st.info("Define segments. Payment re‑calculated at each segment to amortize the remaining balance over the remaining term.")
    default_periods = [
        {"months": min(36, months), "annual_rate": 6.75},
        {"months": max(0, months - min(36, months)), "annual_rate": 9.50},
    ]
    periods_df = st.data_editor(
        pd.DataFrame(default_periods),
        num_rows="dynamic",
        use_container_width=True,
        key="periods_editor",
        column_config={
            "months": st.column_config.NumberColumn("Months", min_value=1, step=1),
            "annual_rate": st.column_config.NumberColumn("Annual Rate (%)", min_value=0.0, step=0.05, format="%.2f"),
        },
    )
    total_defined = int(periods_df["months"].sum()) if not periods_df.empty else 0
    if total_defined != months:
        st.warning(f"Your segments total **{total_defined} months**, but tenure is **{months} months}**.")

    periods = [(int(r.months), float(r.annual_rate)) for r in periods_df.itertuples(index=False)]
    sched = amortize_variable(loan_amt, periods, start_date=start_date)
    eff_rate = solve_effective_rate_from_schedule(loan_amt, sched["Payment"].tolist()) if not sched.empty else 0.0
    meta = {"mode": "Variable", "effective_rate": eff_rate, "emi": float(sched["Payment"].iloc[0]) if not sched.empty else 0.0}

elif mode == "Hybrid (teaser then fixed)":
    teaser_months = st.number_input("Teaser period (months)", min_value=1, max_value=max(2, months-1), value=min(36, months-1), step=1)
    teaser_rate = st.number_input("Teaser Annual Rate (%)", min_value=0.0, value=6.75, step=0.05, format="%.2f")
    post_rate = st.number_input("Post-Teaser Annual Rate (%)", min_value=0.0, value=9.50, step=0.05, format="%.2f")
    sched = amortize_hybrid_teaser_then_fixed(loan_amt, int(teaser_months), teaser_rate, post_rate, months, start_date=start_date)
    eff_rate = solve_effective_rate_from_schedule(loan_amt, sched["Payment"].tolist()) if not sched.empty else 0.0
    meta = {"mode": "Hybrid", "effective_rate": eff_rate, "emi": float(sched["Payment"].iloc[0]) if not sched.empty else 0.0}

elif mode == "Tiered by amount (slabs)":
    st.info("Enter cumulative caps and rates. Example: cap 5,000,000 at 6.75%, cap blank at 9.50% for the rest.")
    default_slabs = [
        {"cap_lkr": 5_000_000, "annual_rate": 6.75},     # up to 5M
        {"cap_lkr": None, "annual_rate": 9.50},          # remainder
    ]
    slabs_df = st.data_editor(
        pd.DataFrame(default_slabs),
        num_rows="dynamic",
        use_container_width=True,
        key="slabs_editor",
        column_config={
            "cap_lkr": st.column_config.NumberColumn("Cumulative cap (LKR)", min_value=0, step=100000),
            "annual_rate": st.column_config.NumberColumn("Annual Rate (%)", min_value=0.0, step=0.05, format="%.2f"),
        },
        hide_index=True
    ).copy()

    slabs_df = slabs_df.sort_values(by=["cap_lkr"], na_position="last")
    tranches = build_slabs_tranches(loan_amt, slabs_df)

    tranche_schedules = []
    for amount, rate in tranches:
        df_t = amortize_fixed(amount, rate, months, start_date=start_date)
        tranche_schedules.append(df_t)

    if tranche_schedules:
        sched = combine_tranche_schedules(tranche_schedules)
        total_emi = float(sched["Payment"].iloc[0]) if not sched.empty else 0.0
        eff_rate = solve_effective_rate_for_emi(loan_amt, months, total_emi) if total_emi > 0 else 0.0
        meta = {"mode": "Tiered (slabs)", "effective_rate": eff_rate, "emi": total_emi}
    else:
        sched = pd.DataFrame()
        meta = {"mode": "Tiered (slabs)", "effective_rate": 0.0, "emi": 0.0}

elif mode == "Multi‑tranche (principal slabs with own schedules)":
    st.info("Define tranche **amounts**, then define a **schedule** for each tranche with rows: tranche id, months, annual rate.")
    # Example defaults: 5M with 36m @6.75% then 9.5%, plus 5M at 9.5% for full term
    default_tranches = [
        {"tranche": "A", "amount": min(5_000_000, loan_amt)},
        {"tranche": "B", "amount": max(0.0, loan_amt - min(5_000_000, loan_amt))}
    ]
    tr_df = st.data_editor(
        pd.DataFrame(default_tranches),
        num_rows="dynamic",
        use_container_width=True,
        key="tranches_editor",
        column_config={
            "tranche": st.column_config.TextColumn("Tranche ID"),
            "amount": st.column_config.NumberColumn("Amount (LKR)", min_value=0.0, step=100000),
        },
        hide_index=True
    ).copy()

    default_sched = [
        {"tranche": "A", "months": min(36, months), "annual_rate": 6.75},
        {"tranche": "A", "months": max(0, months - min(36, months)), "annual_rate": 9.50},
        {"tranche": "B", "months": months, "annual_rate": 9.50},
    ]
    sch_df = st.data_editor(
        pd.DataFrame(default_sched),
        num_rows="dynamic",
        use_container_width=True,
        key="tranche_sched_editor",
        column_config={
            "tranche": st.column_config.TextColumn("Tranche ID"),
            "months": st.column_config.NumberColumn("Months", min_value=1, step=1),
            "annual_rate": st.column_config.NumberColumn("Annual Rate (%)", min_value=0.0, step=0.05, format="%.2f"),
        },
        hide_index=True
    ).copy()

    # Build schedules per tranche
    tranche_schedules = []
    for row in tr_df.itertuples(index=False):
        t_id = str(row.tranche)
        amt = float(row.amount)
        if amt <= 0:
            continue
        segs = sch_df[sch_df["tranche"] == t_id][["months", "annual_rate"]]
        if segs.empty:
            st.warning(f"No schedule rows for tranche '{t_id}'. Skipping.")
            continue
        segs_list = [(int(r.months), float(r.annual_rate)) for r in segs.itertuples(index=False)]
        total_defined = sum(m for m, _ in segs_list)
        if total_defined != months:
            st.warning(f"Tranche '{t_id}' segments total {total_defined} months; tenure is {months}.")
        df_t = amortize_variable(amt, segs_list, start_date=start_date)
        tranche_schedules.append(df_t)

    if tranche_schedules:
        sched = combine_tranche_schedules(tranche_schedules)
        eff_rate = solve_effective_rate_from_schedule(loan_amt, sched["Payment"].tolist()) if not sched.empty else 0.0
        meta = {"mode": "Multi‑tranche", "effective_rate": eff_rate, "emi": float(sched["Payment"].iloc[0]) if not sched.empty else 0.0}
    else:
        sched = pd.DataFrame()
        meta = {"mode": "Multi‑tranche", "effective_rate": 0.0, "emi": 0.0}

else:  # Tiered slabs (per‑slab teaser)
    st.info("Each slab row has its own teaser months/rate, and a post‑teaser rate. Loan is allocated sequentially by cumulative caps.")
    default_slabs2 = [
        {"cap_lkr": 5_000_000, "teaser_months": 36, "teaser_rate": 6.75, "post_rate": 9.50},
        {"cap_lkr": None,      "teaser_months": 0,  "teaser_rate": 0.00, "post_rate": 9.50},
    ]
    slabs2_df = st.data_editor(
        pd.DataFrame(default_slabs2),
        num_rows="dynamic",
        use_container_width=True,
        key="slabs2_editor",
        column_config={
            "cap_lkr": st.column_config.NumberColumn("Cumulative cap (LKR)", min_value=0, step=100000),
            "teaser_months": st.column_config.NumberColumn("Teaser months", min_value=0, step=1),
            "teaser_rate": st.column_config.NumberColumn("Teaser annual %", min_value=0.0, step=0.05, format="%.2f"),
            "post_rate": st.column_config.NumberColumn("Post‑teaser annual %", min_value=0.0, step=0.05, format="%.2f"),
        },
        hide_index=True
    ).copy()

    slabs2_df = slabs2_df.sort_values(by=["cap_lkr"], na_position="last")

    # Build tranches: (amount, teaser_months, teaser_rate, post_rate)
    tranches2 = []
    remaining = loan_amt
    prev_cap = 0.0
    for r in slabs2_df.itertuples(index=False):
        cap = float(r.cap_lkr) if pd.notna(r.cap_lkr) else float("inf")
        band_capacity = max(0.0, cap - prev_cap)
        alloc = min(remaining, band_capacity)
        if alloc > 1e-6:
            tranches2.append((alloc, int(r.teaser_months), float(r.teaser_rate), float(r.post_rate)))
            remaining -= alloc
        prev_cap = cap
        if remaining <= 1e-6:
            break

    if sum(t[0] for t in tranches2) < loan_amt - 1e-6:
        st.warning("Your caps don’t fully cover the loan amount. Add a final row with blank cap for the remainder.")

    tranche_schedules = []
    for amount, tmonths, trate, prate in tranches2:
        if tmonths <= 0:
            df_t = amortize_fixed(amount, prate, months, start_date=start_date)
        else:
            tmonths = min(tmonths, months)
            df_t = amortize_hybrid_teaser_then_fixed(amount, tmonths, trate, prate, months, start_date=start_date)
        tranche_schedules.append(df_t)

    if tranche_schedules:
        sched = combine_tranche_schedules(tranche_schedules)
        eff_rate = solve_effective_rate_from_schedule(loan_amt, sched["Payment"].tolist()) if not sched.empty else 0.0
        meta = {"mode": "Tiered per‑slab teaser", "effective_rate": eff_rate,
                "emi": float(sched["Payment"].iloc[0]) if not sched.empty else 0.0}
    else:
        sched = pd.DataFrame()
        meta = {"mode": "Tiered per‑slab teaser", "effective_rate": 0.0, "emi": 0.0}

# =========================
# Outputs
# =========================

st.markdown("---")
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Loan Amount (LKR)", f"{loan_amt:,.0f}")
with k2:
    st.metric("Tenure (months)", f"{months}")
with k3:
    st.metric("Equivalent Single Rate (annual %)", f"{meta.get('effective_rate', 0.0):.2f}")

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Amortization schedule")
    if not sched.empty:
        view = sched.copy()
        for col in ["Payment", "Interest", "Principal", "Balance"]:
            view[col] = view[col].round(2)
        st.dataframe(view, use_container_width=True, height=420)
    else:
        st.write("No data yet. Adjust inputs on the left.")

with c2:
    st.subheader("Summary")

    def summarize_schedule(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame([{"Total Payment": 0.0, "Total Interest": 0.0, "Tenure (months)": 0, "EMI": 0.0}])
        total_payment = float(df["Payment"].sum())
        total_interest = float(df["Interest"].sum())
        tenure = int(df["Period"].max())
        emi = float(df["Payment"].iloc[0])
        return pd.DataFrame([{
            "Total Payment": total_payment,
            "Total Interest": total_interest,
            "Tenure (months)": tenure,
            "EMI": emi
        }])

    st.table(summarize_schedule(sched).round(2))

st.markdown("#### Balance over time")
if not sched.empty:
    chart_df = sched[["Date", "Balance"]].set_index("Date")
    st.line_chart(chart_df)

# =========================
# Export (in‑memory Excel using xlsxwriter)
# =========================

st.markdown("---")
colx, _ = st.columns([1, 2])
with colx:
    if not sched.empty:
        fn = st.text_input("Export filename", value="loan_schedule.xlsx")

        # Build summary sheet
        summary_df = pd.DataFrame([{
            "Total Payment": float(sched["Payment"].sum()),
            "Total Interest": float(sched["Interest"].sum()),
            "Tenure (months)": int(sched["Period"].max()),
            "EMI": float(sched["Payment"].iloc[0]),
            "Equivalent Single Rate (annual %)": meta.get("effective_rate", 0.0)
        }]).round(2)

        # Create Excel workbook in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            sched.to_excel(writer, index=False, sheet_name="Schedule")
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
        output.seek(0)

        st.download_button(
            label="Download Excel (.xlsx)",
            data=output,
            file_name=fn if fn.lower().endswith(".xlsx") else f"{fn}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# Footer
st.caption("© NSB Tools — Streamlit demo. ")
