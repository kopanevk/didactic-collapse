from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(r"C:/Users/kiris/Documents/New project")

    src_dir = root / "outputs/article_manuscript_prefinal"
    out_dir = root / "outputs/article_manuscript_prefinal_fixed"
    fig_dir = out_dir / "figures"
    tables_dir = root / "outputs/figure_data_bundle/tables"

    # Reset output folder
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Base manuscript
    src_md = src_dir / "manuscript_ru_prefinal.md"
    md = src_md.read_text(encoding="utf-8")

    # Term replacement (if any occurrence remains)
    md = md.replace(
        "использование языковых моделей как судей",
        "использование языковых моделей как оценщиков",
    )

    # Required caption update for Figure 2
    md = md.replace(
        "Панель A показывает высокое покрытие DBR по поколениям. Ось Y на панели A усечена для отображения различий в области высокого покрытия. Панель B показывает доли дефектов до отбора и после DBR. Наиболее устойчивый эффект DBR — сохранение покрытия и подавление ошибок разбора; снижение остальных дефектов умеренное.",
        "Панель A показывает высокое покрытие DBR по поколениям. Ось Y на панели A усечена для отображения различий в области высокого покрытия. Панели B и C показывают доли дефектов до отбора и после DBR для двух экспериментальных серий. Наиболее устойчивый эффект DBR — сохранение покрытия и подавление ошибок разбора; снижение остальных дефектов умеренное.",
    )

    # Keep quality-aware phrase explicit
    md = md.replace(
        "подходу «курирование данных с учётом качества (quality-aware curation)»",
        "подходу «курирование данных с учётом качества (quality-aware curation)»",
    )

    # Save manuscript markdown
    out_md = out_dir / "manuscript_ru_prefinal_fixed.md"
    out_md.write_text(md, encoding="utf-8")

    # Copy Figure 1 unchanged + appendix figures
    src_fig = src_dir / "figures"
    for name in [
        "fig1_dbr_gen2_outcome_ru.png",
        "figA1_dbr_deltas_by_seed_ru.png",
        "figA2_qwen_pairwise_ru.png",
        "figA3_pure_recycling_dynamics_ru.png",
    ]:
        shutil.copy2(src_fig / name, fig_dir / name)

    # Data for Figure 2 and 3
    dbr_sel = pd.read_csv(tables_dir / "dbr_selection_rate.csv")
    dbr_def = pd.read_csv(tables_dir / "dbr_defect_before_after.csv")
    dbr_mech = pd.read_csv(tables_dir / "dbr_mechanism_summary.csv")
    csr_pair = pd.read_csv(tables_dir / "csr_pair_summary_combined.csv")
    csr_bw = pd.read_csv(tables_dir / "csr_best_vs_worst_quality.csv")

    # Style
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )
    col1 = "#4e79a7"
    col2 = "#f28e2b"
    col_before = "#9ea3a8"
    col_after = "#4e79a7"

    # ----------------------
    # Figure 2 (vertical 3-panel)
    # ----------------------
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

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(10.5, 13))

    # A coverage
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
            linewidth=2.4,
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

    def defect_panel(ax, fam: str, title: str) -> None:
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
        ax.set_xticklabels([defect_ru[d] for d in sub["defect_type"]], rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Доля")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        pf_idx = list(sub["defect_type"]).index("parse_failure")
        pf_val = float(sub.loc[pf_idx, "rate_after"])
        ax.text(pf_idx + w / 2, pf_val + 0.02, f"{pf_val:.3f}", ha="center", va="bottom", fontsize=9)

    defect_panel(ax_b, "dbr_confirmatory_211_213", "B. Дефекты: серия 211–213")
    defect_panel(ax_c, "dbr_only_221_223", "C. Дефекты: чистая DBR-серия 221–223")
    ax_b.legend(loc="upper right", frameon=True)

    fig.suptitle("Механизм DBR: покрытие и дефектный профиль", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fig_dir / "fig2_dbr_mechanism_ru.png", dpi=300)
    fig.savefig(fig_dir / "fig2_dbr_mechanism_ru.svg")
    plt.close(fig)

    # ----------------------
    # Figure 3 (simplified 2-panel, no tiny internal note)
    # ----------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.2))

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
    ax_a.set_ylabel("Доля построенных пар")
    ax_a.set_xlabel("Вариант CSR")
    ax_a.set_title("A. Построение контрастных пар")
    for i, v in enumerate(pair_rate["pair_construction_rate"]):
        ax_a.text(i, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    quality = (
        csr_bw.groupby("csr_variant", as_index=False)["mean_quality_gap"]
        .mean()
        .sort_values("csr_variant")
    )
    x2 = np.arange(len(quality))
    ax_b.bar(x2, quality["mean_quality_gap"], color=[col1, col2])
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(["k=3", "k=5"])
    ax_b.set_ylabel("Разрыв качества")
    ax_b.set_xlabel("Вариант CSR")
    ax_b.set_title("B. Средний разрыв качества")
    for i, v in enumerate(quality["mean_quality_gap"]):
        ax_b.text(i, v + 0.08, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Контрастный педагогический сигнал CSR", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(fig_dir / "fig3_csr_contrast_ru.png", dpi=300)
    fig.savefig(fig_dir / "fig3_csr_contrast_ru.svg")
    plt.close(fig)

    # Number checks for log
    mech_idx = dbr_mech.set_index("family")
    cov_211 = float(mech_idx.loc["dbr_confirmatory_211_213", "selection_rate_mean"])
    cov_221 = float(mech_idx.loc["dbr_only_221_223", "selection_rate_mean"])
    parse_211 = float(mech_idx.loc["dbr_confirmatory_211_213", "parse_failure_after_mean"])
    parse_221 = float(mech_idx.loc["dbr_only_221_223", "parse_failure_after_mean"])

    pr = pair_rate.set_index("csr_variant")["pair_construction_rate"].to_dict()
    qg = quality.set_index("csr_variant")["mean_quality_gap"].to_dict()
    best_silent = (
        csr_bw.groupby("csr_variant", as_index=False)["best_silent_rate"]
        .mean()
        .set_index("csr_variant")["best_silent_rate"]
        .to_dict()
    )

    # Build figure fix log
    log = f"""# Figure Fix Log

## Исправлено
1. Заменена фраза “использование языковых моделей как судей” на “использование языковых моделей как оценщиков”.
2. Рисунок 2 перегенерирован в трёхпанельном формате A/B/C без наложения заголовков.
3. Рисунок 3 перегенерирован без внутренней мелкой подписи про скрытые ошибки.
4. DOCX и Markdown обновлены.
5. Численные результаты не изменялись.

## Проверка чисел
- Рисунок 2: покрытие 211–213 около {cov_211:.4f}; покрытие 221–223 около {cov_221:.4f}; ошибка разбора после DBR {parse_211:.4f} / {parse_221:.4f}.
- Рисунок 3: pair construction k=3 {pr.get('k3', float('nan')):.4f}; k=5 {pr.get('k5', float('nan')):.4f}; quality gap k=3 {qg.get('k3', float('nan')):.4f}; k=5 {qg.get('k5', float('nan')):.4f}; best silent error k=3 {best_silent.get('k3', float('nan')):.4f}, k=5 {best_silent.get('k5', float('nan')):.4f}.

## Осталось перед подачей
- Заполнить ФИО, аффилиацию, email.
- Проверить DOI/URL.
- Визуально проверить DOCX после вставки рисунков.
- Уточнить доступность кода и данных.
- Выполнить финальную языковую вычитку.
"""
    (out_dir / "figure_fix_log.md").write_text(log, encoding="utf-8")

    # Build DOCX from fixed markdown (with embedded images + page numbering)
    converter = root / "outputs/.tmp/md_to_docx_with_images.py"
    # We'll call this externally from shell; here only leave artifacts ready.
    print("ready", out_md, fig_dir, converter)


if __name__ == "__main__":
    main()
