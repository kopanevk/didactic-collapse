from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(r"C:/Users/kiris/Documents/New project")
in_dir = root / "outputs" / "figure_data_bundle" / "tables"
out_base = root / "outputs" / "article_figures_final"
main_dir = out_base / "main"
app_dir = out_base / "appendix"

for d in [main_dir, app_dir]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    }
)

COL_SERIES_1 = "#4e79a7"
COL_SERIES_2 = "#f28e2b"
COL_NEUTRAL = "#8c8c8c"
COL_BEFORE = "#9ea3a8"
COL_AFTER = "#4e79a7"

family_ru = {
    "dbr_confirmatory_211_213": "Серия 211–213",
    "dbr_only_221_223": "Чистая DBR-серия 221–223",
}

metric_ru = {
    "delta_accuracy": "Точность",
    "delta_pedagogy": "Педагогическая оценка",
    "delta_silent": "Доля скрытых ошибок",
}

defect_ru = {
    "incorrect_answer": "Неправильный ответ",
    "low_reasoning": "Слабое рассуждение",
    "low_structure": "Слабая структура",
    "silent_error": "Скрытая ошибка",
    "parse_failure": "Ошибка разбора",
}

dbr_sum = pd.read_csv(in_dir / "dbr_gen2_summary.csv")
dbr_seed = pd.read_csv(in_dir / "dbr_gen2_by_seed.csv")
dbr_sel = pd.read_csv(in_dir / "dbr_selection_rate.csv")
dbr_def = pd.read_csv(in_dir / "dbr_defect_before_after.csv")
dbr_mech = pd.read_csv(in_dir / "dbr_mechanism_summary.csv")
csr_pair = pd.read_csv(in_dir / "csr_pair_summary_combined.csv")
csr_bw = pd.read_csv(in_dir / "csr_best_vs_worst_quality.csv")
qsum = pd.read_csv(in_dir / "qwen_dbr_pairwise_summary.csv")
qseed = pd.read_csv(in_dir / "qwen_dbr_pairwise_by_seed.csv")
collapse = pd.read_csv(in_dir / "collapse_evidence_pure_by_seed_generation.csv")

log_lines = []
log_lines.append("# Журнал генерации рисунков\n")
log_lines.append("Источник таблиц: outputs/figure_data_bundle/tables/\n")
log_lines.append("Новые эксперименты и оценочные вызовы не запускались.\n")

# Figure 1
fig, ax = plt.subplots(figsize=(8.0, 5.2))
metrics = ["delta_accuracy", "delta_pedagogy", "delta_silent"]
x = np.arange(len(metrics))
width = 0.34
for i, fam in enumerate(["dbr_confirmatory_211_213", "dbr_only_221_223"]):
    sub = dbr_sum[dbr_sum["family"] == fam].set_index("metric")
    means = np.array([sub.loc[m, "mean_delta"] for m in metrics], dtype=float)
    lows = np.array([sub.loc[m, "ci_low"] for m in metrics], dtype=float)
    highs = np.array([sub.loc[m, "ci_high"] for m in metrics], dtype=float)
    yerr = np.vstack([means - lows, highs - means])
    color = COL_SERIES_1 if i == 0 else COL_SERIES_2
    ax.bar(
        x + (i - 0.5) * width,
        means,
        width=width,
        color=color,
        label=family_ru[fam],
        yerr=yerr,
        capsize=4,
    )

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([metric_ru[m] for m in metrics])
ax.set_ylabel("Дельта DBR − обычное рециклирование")
ax.set_title("Итоговые эффекты DBR на втором поколении")
ax.legend(loc="best", frameon=True)
fig.tight_layout()
fig.savefig(main_dir / "fig1_dbr_gen2_outcome_ru.png", dpi=300)
fig.savefig(main_dir / "fig1_dbr_gen2_outcome_ru.svg")
plt.close(fig)

checks_fig1 = {
    ("dbr_confirmatory_211_213", "delta_accuracy"): 0.0625,
    ("dbr_confirmatory_211_213", "delta_pedagogy"): 0.1461,
    ("dbr_confirmatory_211_213", "delta_silent"): -0.0636,
    ("dbr_only_221_223", "delta_accuracy"): 0.0333,
    ("dbr_only_221_223", "delta_pedagogy"): 0.0200,
    ("dbr_only_221_223", "delta_silent"): 0.0267,
}
v = dbr_sum.set_index(["family", "metric"])["mean_delta"]
log_lines.append("\n## Проверка Рисунка 1\n")
for k, exp in checks_fig1.items():
    got = float(v.loc[k])
    ok = abs(got - exp) < 0.002
    log_lines.append(
        f"- {k[0]} / {k[1]}: получено {got:.4f}, ожидается {exp:.4f}, статус: {'OK' if ok else 'РАСХОЖДЕНИЕ'}"
    )

# Figure 2 (no heatmap)
fig = plt.figure(figsize=(11.0, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.45])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

for fam, color in [
    ("dbr_confirmatory_211_213", COL_SERIES_1),
    ("dbr_only_221_223", COL_SERIES_2),
]:
    grp = (
        dbr_sel[dbr_sel["family"] == fam]
        .groupby("generation", as_index=False)["selection_rate"]
        .mean()
    )
    axA.plot(
        grp["generation"],
        grp["selection_rate"],
        marker="o",
        linewidth=2,
        color=color,
        label=family_ru[fam],
    )

axA.set_xticks([0, 1, 2])
axA.set_xticklabels(["Gen0", "Gen1", "Gen2"])
axA.set_ylim(0.90, 1.00)
axA.set_xlabel("Поколение")
axA.set_ylabel("Покрытие")
axA.set_title("A. Покрытие DBR")
axA.grid(True, alpha=0.25)
axA.legend(loc="lower left", frameon=True)

order = ["incorrect_answer", "low_reasoning", "low_structure", "silent_error", "parse_failure"]
base_pos_1 = np.arange(len(order))
base_pos_2 = base_pos_1 + len(order) + 1.2
w = 0.35

for fam, pos in [
    ("dbr_confirmatory_211_213", base_pos_1),
    ("dbr_only_221_223", base_pos_2),
]:
    sub = (
        dbr_def[dbr_def["family"] == fam]
        .groupby("defect_type", as_index=False)[["rate_before", "rate_after"]]
        .mean()
    )
    sub = sub.set_index("defect_type").reindex(order).reset_index()
    axB.bar(
        pos - w / 2,
        sub["rate_before"],
        width=w,
        color=COL_BEFORE,
        label="До отбора" if fam == "dbr_confirmatory_211_213" else None,
    )
    axB.bar(
        pos + w / 2,
        sub["rate_after"],
        width=w,
        color=COL_AFTER,
        label="После DBR" if fam == "dbr_confirmatory_211_213" else None,
    )
    pf_val = float(sub[sub["defect_type"] == "parse_failure"]["rate_after"].iloc[0])
    pf_pos = pos[list(sub["defect_type"]).index("parse_failure")]
    axB.text(pf_pos + w / 2, pf_val + 0.015, f"{pf_val:.3f}", ha="center", va="bottom", fontsize=8)

xticks = list(base_pos_1) + list(base_pos_2)
xticklabels = [defect_ru[d] for d in order] + [defect_ru[d] for d in order]
axB.set_xticks(xticks)
axB.set_xticklabels(xticklabels, rotation=28, ha="right")
axB.set_ylim(0, 1.0)
axB.set_ylabel("Доля дефекта")
axB.set_title("B. Дефекты: до и после DBR")
axB.grid(True, axis="y", alpha=0.25)
axB.legend(loc="upper right", frameon=True)
axB.text(
    base_pos_1.mean(),
    1.02,
    family_ru["dbr_confirmatory_211_213"],
    ha="center",
    va="bottom",
    fontsize=9,
    transform=axB.get_xaxis_transform(),
)
axB.text(
    base_pos_2.mean(),
    1.02,
    family_ru["dbr_only_221_223"],
    ha="center",
    va="bottom",
    fontsize=9,
    transform=axB.get_xaxis_transform(),
)

fig.suptitle("Механизм DBR: покрытие и дефектный профиль", y=1.02)
fig.tight_layout()
fig.savefig(main_dir / "fig2_dbr_mechanism_ru.png", dpi=300, bbox_inches="tight")
fig.savefig(main_dir / "fig2_dbr_mechanism_ru.svg", bbox_inches="tight")
plt.close(fig)

log_lines.append("\n## Проверка Рисунка 2\n")
mech_idx = dbr_mech.set_index("family")
sel_211 = float(mech_idx.loc["dbr_confirmatory_211_213", "selection_rate_mean"])
sel_221 = float(mech_idx.loc["dbr_only_221_223", "selection_rate_mean"])
pf_211 = float(mech_idx.loc["dbr_confirmatory_211_213", "parse_failure_after_mean"])
pf_221 = float(mech_idx.loc["dbr_only_221_223", "parse_failure_after_mean"])
log_lines.append(
    f"- Среднее покрытие 211–213: {sel_211:.4f} (ожидалось ~0.9756) -> {'OK' if abs(sel_211-0.9756)<0.002 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Среднее покрытие 221–223: {sel_221:.4f} (ожидалось ~0.9656) -> {'OK' if abs(sel_221-0.9656)<0.002 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Ошибка разбора после DBR (211–213): {pf_211:.4f} (ожидалось 0.0000) -> {'OK' if abs(pf_211)<1e-9 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Ошибка разбора после DBR (221–223): {pf_221:.4f} (ожидалось 0.0000) -> {'OK' if abs(pf_221)<1e-9 else 'РАСХОЖДЕНИЕ'}"
)

# Figure 3
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 5.0))
pair_rate = (
    csr_pair.groupby("csr_variant", as_index=False)["pair_construction_rate"].mean().sort_values("csr_variant")
)
xv = np.arange(len(pair_rate))
axA.bar(xv, pair_rate["pair_construction_rate"], color=[COL_SERIES_1, COL_SERIES_2])
axA.set_xticks(xv)
axA.set_xticklabels(["k=3", "k=5"])
axA.set_ylim(0, 0.8)
axA.set_ylabel("Доля построенных пар")
axA.set_xlabel("Вариант CSR")
axA.set_title("A. Доля построенных контрастных пар")
for i, val in enumerate(pair_rate["pair_construction_rate"]):
    axA.text(i, val + 0.015, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

quality = (
    csr_bw.groupby("csr_variant", as_index=False)["mean_quality_gap"].mean().sort_values("csr_variant")
)
best_silent = (
    csr_bw.groupby("csr_variant", as_index=False)["best_silent_rate"].mean().sort_values("csr_variant")
)
axB.bar(np.arange(len(quality)), quality["mean_quality_gap"], color=[COL_SERIES_1, COL_SERIES_2])
axB.set_xticks(np.arange(len(quality)))
axB.set_xticklabels(["k=3", "k=5"])
axB.set_ylabel("Средний разрыв качества (best − worst)")
axB.set_xlabel("Вариант CSR")
axB.set_title("B. Контраст качества")
for i, val in enumerate(quality["mean_quality_gap"]):
    axB.text(i, val + 0.08, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
axB.text(
    0.5,
    0.02,
    "Скрытые ошибки у лучших ответов: 0.000 для обоих вариантов",
    ha="center",
    va="bottom",
    transform=axB.transAxes,
    fontsize=8,
)

fig.suptitle("Контрастный педагогический сигнал CSR", y=1.02)
fig.tight_layout()
fig.savefig(main_dir / "fig3_csr_contrast_ru.png", dpi=300, bbox_inches="tight")
fig.savefig(main_dir / "fig3_csr_contrast_ru.svg", bbox_inches="tight")
plt.close(fig)

log_lines.append("\n## Проверка Рисунка 3\n")
pr = dict(zip(pair_rate["csr_variant"], pair_rate["pair_construction_rate"]))
qg = dict(zip(quality["csr_variant"], quality["mean_quality_gap"]))
bs = dict(zip(best_silent["csr_variant"], best_silent["best_silent_rate"]))
log_lines.append(
    f"- Pair construction k=3: {pr.get('k3', np.nan):.4f} (ожидалось 0.5111) -> {'OK' if abs(pr.get('k3',0)-0.5111)<0.002 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Pair construction k=5: {pr.get('k5', np.nan):.4f} (ожидалось 0.5889) -> {'OK' if abs(pr.get('k5',0)-0.5889)<0.002 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Quality gap k=3: {qg.get('k3', np.nan):.4f} (ожидалось 5.3968) -> {'OK' if abs(qg.get('k3',0)-5.3968)<0.01 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Quality gap k=5: {qg.get('k5', np.nan):.4f} (ожидалось 5.8029) -> {'OK' if abs(qg.get('k5',0)-5.8029)<0.01 else 'РАСХОЖДЕНИЕ'}"
)
log_lines.append(
    f"- Best silent k=3: {bs.get('k3', np.nan):.4f}; k=5: {bs.get('k5', np.nan):.4f} (ожидалось 0.0000) -> {'OK' if abs(bs.get('k3',1))<1e-9 and abs(bs.get('k5',1))<1e-9 else 'РАСХОЖДЕНИЕ'}"
)

# Appendix A1
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharex=True)
seed_plot = dbr_seed.copy()
for ax, col, title in zip(
    axes,
    ["delta_accuracy", "delta_pedagogy", "delta_silent"],
    ["Дельта точности", "Дельта педагогической оценки", "Дельта доли скрытых ошибок"],
):
    for fam, color in [
        ("dbr_confirmatory_211_213", COL_SERIES_1),
        ("dbr_only_221_223", COL_SERIES_2),
    ]:
        grp = seed_plot[seed_plot["family"] == fam].sort_values("seed")
        ax.plot(grp["seed"], grp[col], marker="o", linewidth=2, color=color, label=family_ru[fam])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Запуск")
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel("Дельта DBR − обычное рециклирование")
axes[-1].legend(loc="best", fontsize=8)
fig.suptitle("Рисунок A1. Дельты DBR по отдельным запускам", y=1.03)
fig.tight_layout()
fig.savefig(app_dir / "figA1_dbr_deltas_by_seed_ru.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Appendix A2
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.8, 4.8))
row = qsum.iloc[0]
counts = [row["dbr_wins"], row["pure_wins"], row["ties"]]
labels = ["Победы DBR", "Победы обычного\nрециклирования", "Ничьи"]
axL.bar(np.arange(3), counts, color=[COL_SERIES_1, COL_SERIES_2, COL_NEUTRAL])
axL.set_xticks(np.arange(3))
axL.set_xticklabels(labels)
axL.set_ylabel("Количество пар")
axL.set_title("A. Итоги парной проверки (n=39 успешных)")
for i, val in enumerate(counts):
    axL.text(i, val + 0.4, f"{int(val)}", ha="center", va="bottom", fontsize=9)

qseed_sorted = qseed.sort_values("seed")
axR.plot(qseed_sorted["seed"], qseed_sorted["dbr_win_rate"], marker="o", color=COL_SERIES_1, label="DBR")
axR.plot(
    qseed_sorted["seed"],
    qseed_sorted["pure_win_rate"],
    marker="o",
    color=COL_SERIES_2,
    label="Обычное рециклирование",
)
axR.plot(qseed_sorted["seed"], qseed_sorted["tie_rate"], marker="o", color=COL_NEUTRAL, label="Ничьи")
axR.set_ylim(0, 1)
axR.set_xlabel("Запуск")
axR.set_ylabel("Доля")
axR.set_title("B. Доли исходов по запускам")
axR.grid(True, alpha=0.25)
axR.legend(loc="best", fontsize=8)
fig.suptitle("Дополнительная парная проверка Qwen3-235B", y=1.03)
fig.tight_layout()
fig.savefig(app_dir / "figA2_qwen_pairwise_ru.png", dpi=300, bbox_inches="tight")
plt.close(fig)

log_lines.append("\n## Проверка Рисунка A2\n")
checks_a2 = {
    "selected_pairs": 48,
    "successful_pairs": 39,
    "dbr_wins": 13,
    "pure_wins": 8,
    "ties": 18,
    "margin": 0.1282,
}
for key, exp in checks_a2.items():
    got = float(row[key])
    tol = 0.001 if key == "margin" else 0.1
    ok = abs(got - exp) < tol
    fmt = "{:.4f}" if key == "margin" else "{:.0f}"
    log_lines.append(
        f"- {key}: получено {fmt.format(got)}, ожидается {fmt.format(exp)}, статус: {'OK' if ok else 'РАСХОЖДЕНИЕ'}"
    )

# Appendix A3
fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8), sharex=True)
for fam, color in [
    ("dbr_confirmatory_211_213", COL_SERIES_1),
    ("dbr_only_221_223", COL_SERIES_2),
]:
    sub = (
        collapse[collapse["experiment_family"] == fam]
        .groupby("generation", as_index=False)[["accuracy_mean", "pedagogical_score_mean", "silent_error_rate"]]
        .mean()
    )
    axes[0].plot(sub["generation"], sub["accuracy_mean"], marker="o", linewidth=2, color=color, label=family_ru[fam])
    axes[1].plot(
        sub["generation"],
        sub["pedagogical_score_mean"],
        marker="o",
        linewidth=2,
        color=color,
        label=family_ru[fam],
    )
    axes[2].plot(
        sub["generation"], sub["silent_error_rate"], marker="o", linewidth=2, color=color, label=family_ru[fam]
    )

for ax, title in zip(axes, ["Точность", "Педагогическая оценка", "Доля скрытых ошибок"]):
    ax.set_title(title)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Gen0", "Gen1", "Gen2"])
    ax.set_xlabel("Поколение")
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel("Среднее значение")
axes[-1].legend(loc="best", fontsize=8)
fig.suptitle("Обычное рециклирование: межпоколенческая динамика", y=1.03)
fig.tight_layout()
fig.savefig(app_dir / "figA3_pure_recycling_dynamics_ru.png", dpi=300, bbox_inches="tight")
plt.close(fig)

captions = """# Подписи к рисункам (русская версия)

## Основная статья

**Рисунок 1. Итоговые эффекты DBR на втором поколении.**
Показаны средние дельты DBR − обычное рециклирование по трём запускам. Для точности и педагогической оценки положительная дельта означает улучшение; для доли скрытых ошибок отрицательная дельта означает улучшение. Усики показывают 95% доверительные интервалы по трём запускам. Широкие интервалы подчёркивают смешанный характер итоговых эффектов.

**Рисунок 2. Механизм DBR: покрытие и дефектный профиль.**
Панель A показывает высокое покрытие DBR по поколениям. Ось Y на панели A усечена для отображения различий в области высокого покрытия. Панель B показывает доли дефектов до отбора и после DBR. Наиболее устойчивый эффект DBR — сохранение покрытия и подавление ошибок разбора; снижение остальных дефектов умеренное.

**Рисунок 3. Контрастный педагогический сигнал CSR.**
CSR используется как диагностический анализ, а не как основной итоговый метод. Доля построенных контрастных пар составляет 0.5111 при k=3 и 0.5889 при k=5; средний разрыв качества между лучшим и худшим объяснением — около 5.4–5.8 балла. Это показывает, что одна и та же модель может порождать резко различающиеся по педагогическому качеству объяснения для одной задачи.

## Приложение

**Рисунок A1. Дельты DBR по отдельным запускам.**
Рисунок показывает чувствительность итоговых эффектов DBR к запуску. Для доли скрытых ошибок отрицательная дельта означает улучшение.

**Рисунок A2. Дополнительная парная проверка Qwen3-235B.**
Из 48 выбранных пар успешно оценены 39; DBR выбран 13 раз, обычное рециклирование — 8 раз, ничья — 18 раз. Высокая доля ничьих подчёркивает, что это проверка чувствительности, а не основное доказательство.

**Рисунок A3. Обычное рециклирование: межпоколенческая динамика.**
Рисунок не демонстрирует сильную монотонную форму коллапса; он показывает ненулевую дефектную нагрузку и нестабильность педагогических метрик.

## Пояснение по интерпретации доли скрытых ошибок
Для доли скрытых ошибок отрицательная дельта означает улучшение.
"""
(out_base / "figure_captions_ru.md").write_text(captions, encoding="utf-8")

mismatch_lines = [line for line in log_lines if "РАСХОЖДЕНИЕ" in line]
log_lines.append("\n## Итого по проверке\n")
if mismatch_lines:
    log_lines.append("Обнаружены расхождения. Перед вставкой рисунков нужна ручная проверка.")
else:
    log_lines.append("Численные проверки пройдены, расхождений не обнаружено.")

(out_base / "figure_generation_log.md").write_text("\n".join(log_lines), encoding="utf-8")

print("Generated files in", out_base)
print("Mismatch lines:", len(mismatch_lines))
