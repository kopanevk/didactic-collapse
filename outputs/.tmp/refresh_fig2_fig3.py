from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(r"C:/Users/kiris/Documents/New project")
    tables = root / "outputs/figure_data_bundle/tables"
    out_main = root / "outputs/article_figures_final/main"
    out_main.mkdir(parents=True, exist_ok=True)

    dbr_sel = pd.read_csv(tables / "dbr_selection_rate.csv")
    dbr_def = pd.read_csv(tables / "dbr_defect_before_after.csv")
    csr_pair = pd.read_csv(tables / "csr_pair_summary_combined.csv")
    csr_bw = pd.read_csv(tables / "csr_best_vs_worst_quality.csv")

    family_ru = {
        "dbr_confirmatory_211_213": "Серия 211–213",
        "dbr_only_221_223": "Чистая DBR-серия 221–223",
    }
    defect_order = ["incorrect_answer", "low_reasoning", "low_structure", "silent_error", "parse_failure"]
    defect_ru = {
        "incorrect_answer": "Неправильный\nответ",
        "low_reasoning": "Слабое\nрассуждение",
        "low_structure": "Слабая\nструктура",
        "silent_error": "Скрытая\nошибка",
        "parse_failure": "Ошибка\nразбора",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )
    col1 = "#4e79a7"
    col2 = "#f28e2b"
    col_before = "#9ea3a8"
    col_after = "#4e79a7"

    # -------------------------
    # Figure 2 (three panels)
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_a, ax_b, ax_c = axes

    # Panel A: coverage by generation
    for fam, color in [("dbr_confirmatory_211_213", col1), ("dbr_only_221_223", col2)]:
        grp = (
            dbr_sel[dbr_sel["family"] == fam]
            .groupby("generation", as_index=False)["selection_rate"]
            .mean()
        )
        ax_a.plot(
            grp["generation"],
            grp["selection_rate"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=family_ru[fam],
        )
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(["Gen0", "Gen1", "Gen2"])
    ax_a.set_ylim(0.90, 1.00)
    ax_a.set_xlabel("Поколение")
    ax_a.set_ylabel("Покрытие")
    ax_a.set_title("A. Покрытие DBR по поколениям")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="lower left", frameon=True)

    def _defect_panel(ax, fam: str, title: str) -> None:
        sub = (
            dbr_def[dbr_def["family"] == fam]
            .groupby("defect_type", as_index=False)[["rate_before", "rate_after"]]
            .mean()
        )
        sub = sub.set_index("defect_type").reindex(defect_order).reset_index()
        x = np.arange(len(sub))
        w = 0.36
        ax.bar(x - w / 2, sub["rate_before"], width=w, color=col_before, label="До отбора")
        ax.bar(x + w / 2, sub["rate_after"], width=w, color=col_after, label="После DBR")
        ax.set_xticks(x)
        ax.set_xticklabels([defect_ru[d] for d in sub["defect_type"]])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Доля дефекта")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        pf_idx = list(sub["defect_type"]).index("parse_failure")
        pf_val = float(sub.loc[pf_idx, "rate_after"])
        ax.text(pf_idx + w / 2, pf_val + 0.02, f"{pf_val:.3f}", ha="center", va="bottom", fontsize=8)

    _defect_panel(ax_b, "dbr_confirmatory_211_213", "B. Дефекты до/после DBR:\nсерия 211–213")
    _defect_panel(ax_c, "dbr_only_221_223", "C. Дефекты до/после DBR:\nчистая DBR-серия 221–223")
    ax_b.legend(loc="upper right", frameon=True)

    fig.suptitle("Механизм DBR: покрытие и дефектный профиль", y=1.02)
    fig.tight_layout()
    fig.savefig(out_main / "fig2_dbr_mechanism_ru.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_main / "fig2_dbr_mechanism_ru.svg", bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Figure 3 (simplified)
    # -------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))

    pair_rate = (
        csr_pair.groupby("csr_variant", as_index=False)["pair_construction_rate"]
        .mean()
        .sort_values("csr_variant")
    )
    x = np.arange(len(pair_rate))
    ax_a.bar(x, pair_rate["pair_construction_rate"], color=[col1, col2])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["k=3", "k=5"])
    ax_a.set_ylim(0, 0.8)
    ax_a.set_xlabel("Вариант CSR")
    ax_a.set_ylabel("Доля построенных пар")
    ax_a.set_title("A. Доля построенных контрастных пар")
    for i, v in enumerate(pair_rate["pair_construction_rate"]):
        ax_a.text(i, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    quality = (
        csr_bw.groupby("csr_variant", as_index=False)["mean_quality_gap"]
        .mean()
        .sort_values("csr_variant")
    )
    x2 = np.arange(len(quality))
    ax_b.bar(x2, quality["mean_quality_gap"], color=[col1, col2])
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(["k=3", "k=5"])
    ax_b.set_xlabel("Вариант CSR")
    ax_b.set_ylabel("Средний разрыв качества (best − worst)")
    ax_b.set_title("B. Средний разрыв качества best − worst")
    for i, v in enumerate(quality["mean_quality_gap"]):
        ax_b.text(i, v + 0.08, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Контрастный педагогический сигнал CSR", y=1.02)
    fig.tight_layout()
    fig.savefig(out_main / "fig3_csr_contrast_ru.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_main / "fig3_csr_contrast_ru.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
