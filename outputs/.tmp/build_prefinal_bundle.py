from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(r"C:/Users/kiris/Documents/New project")

    src_md = root / "outputs/article_manuscript_final_polished/manuscript_ru_polished.md"
    src_docx = root / "outputs/article_manuscript_final_polished/manuscript_ru_polished.docx"
    src_claim = root / "outputs/article_manuscript_final_polished/appendix_claim_status.md"
    src_tables = root / "outputs/article_manuscript_final_polished/appendix_tables.md"
    src_checklist = root / "outputs/article_manuscript_final_polished/submission_checklist.md"
    src_metadata = root / "outputs/article_manuscript_final_polished/submission_metadata.md"
    src_change = root / "outputs/article_manuscript_final_polished/theory_change_log.md"
    src_repro = root / "outputs/article_manuscript_final/appendix_reproducibility.md"

    fig_src = root / "outputs/article_figures_final"
    fig_main = fig_src / "main"
    fig_app = fig_src / "appendix"

    out = root / "outputs/article_manuscript_prefinal"
    fig_out = out / "figures"

    if out.exists():
        for p in out.glob("*"):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    text = src_md.read_text(encoding="utf-8")
    lower = text.lower()
    # Selection criteria (fresh + theory-refined)
    if not ("риск дидактического коллапса" in lower[:500]):
        raise RuntimeError("Базовая рукопись не содержит требуемого заголовка с «Риск дидактического коллапса».")
    if not ("сильная форма" in lower and "слабая форма" in lower):
        raise RuntimeError("Базовая рукопись не содержит различение сильной/слабой формы.")
    for n in ["0.30", "0.2708", "0.27", "0.23", "0.19"]:
        if n not in text:
            raise RuntimeError(f"Базовая рукопись не содержит требуемое число: {n}")
    if "модель-судья" in lower:
        raise RuntimeError("В базовой рукописи остался термин «модель-судья».")

    # A2: enforce integer seed ticks 211/212/213
    qsum = pd.read_csv(root / "outputs/figure_data_bundle/tables/qwen_dbr_pairwise_summary.csv")
    qseed = pd.read_csv(root / "outputs/figure_data_bundle/tables/qwen_dbr_pairwise_by_seed.csv").sort_values("seed")

    COL_SERIES_1 = "#4e79a7"
    COL_SERIES_2 = "#f28e2b"
    COL_NEUTRAL = "#8c8c8c"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.8, 4.8))
    row = qsum.iloc[0]
    counts = [row["dbr_wins"], row["pure_wins"], row["ties"]]
    labels = ["Победы DBR", "Победы обычного\nрециклирования", "Ничьи"]
    ax_l.bar(np.arange(3), counts, color=[COL_SERIES_1, COL_SERIES_2, COL_NEUTRAL])
    ax_l.set_xticks(np.arange(3))
    ax_l.set_xticklabels(labels)
    ax_l.set_ylabel("Количество пар")
    ax_l.set_title("A. Итоги парной проверки (n=39 успешных)")
    for i, v in enumerate(counts):
        ax_l.text(i, v + 0.4, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    ax_r.plot(qseed["seed"], qseed["dbr_win_rate"], marker="o", color=COL_SERIES_1, label="DBR")
    ax_r.plot(qseed["seed"], qseed["pure_win_rate"], marker="o", color=COL_SERIES_2, label="Обычное рециклирование")
    ax_r.plot(qseed["seed"], qseed["tie_rate"], marker="o", color=COL_NEUTRAL, label="Ничьи")
    ax_r.set_ylim(0, 1)
    ax_r.set_xlabel("Запуск")
    ax_r.set_ylabel("Доля")
    ax_r.set_title("B. Доли исходов по запускам")
    ax_r.set_xticks([211, 212, 213])
    ax_r.set_xticklabels(["211", "212", "213"])
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(loc="best", fontsize=8)

    fig.suptitle("Дополнительная парная проверка Qwen3-235B", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_app / "figA2_qwen_pairwise_ru.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Copy figures
    fig_map = {
        "fig1_dbr_gen2_outcome_ru.png": fig_main / "fig1_dbr_gen2_outcome_ru.png",
        "fig2_dbr_mechanism_ru.png": fig_main / "fig2_dbr_mechanism_ru.png",
        "fig3_csr_contrast_ru.png": fig_main / "fig3_csr_contrast_ru.png",
        "figA1_dbr_deltas_by_seed_ru.png": fig_app / "figA1_dbr_deltas_by_seed_ru.png",
        "figA2_qwen_pairwise_ru.png": fig_app / "figA2_qwen_pairwise_ru.png",
        "figA3_pure_recycling_dynamics_ru.png": fig_app / "figA3_pure_recycling_dynamics_ru.png",
    }
    missing_figs: list[str] = []
    for name, src in fig_map.items():
        dst = fig_out / name
        if src.exists() and src.stat().st_size > 0:
            shutil.copy2(src, dst)
        else:
            missing_figs.append(name)

    # Edit manuscript: remove old figure placeholder section and insert figures in target sections
    lines = src_md.read_text(encoding="utf-8").splitlines()
    start_75 = next((i for i, l in enumerate(lines) if l.strip().startswith("### 7.5.")), None)
    start_8 = next((i for i, l in enumerate(lines) if l.strip().startswith("## 8.")), None)
    if start_75 is not None and start_8 is not None and start_75 < start_8:
        lines = lines[:start_75] + lines[start_8:]
    text2 = "\n".join(lines)

    fig2_block = """

Рисунок 2 иллюстрирует механизм DBR и подчеркивает, что устойчивый эффект связан прежде всего с контролем дефектного профиля.

![Рисунок 2. Механизм DBR: покрытие и дефектный профиль](figures/fig2_dbr_mechanism_ru.png)

**Рисунок 2. Механизм DBR: покрытие и дефектный профиль.**
Панель A показывает высокое покрытие DBR по поколениям. Ось Y на панели A усечена для отображения различий в области высокого покрытия. Панель B показывает доли дефектов до отбора и после DBR. Наиболее устойчивый эффект DBR — сохранение покрытия и подавление ошибок разбора; снижение остальных дефектов умеренное.
""".strip(
        "\n"
    )

    fig1_block = """

На рисунке 1 показаны итоговые дельты DBR на втором поколении по двум ключевым экспериментальным сериям.

![Рисунок 1. Итоговые эффекты DBR на втором поколении](figures/fig1_dbr_gen2_outcome_ru.png)

**Рисунок 1. Итоговые эффекты DBR на втором поколении.**
Показаны средние дельты DBR − обычное рециклирование по трём запускам. Для точности и педагогической оценки положительная дельта означает улучшение; для доли скрытых ошибок отрицательная дельта означает улучшение. Усики показывают 95% доверительные интервалы по трём запускам. Широкие интервалы подчёркивают смешанный характер итоговых эффектов.
""".strip(
        "\n"
    )

    fig3_block = """

Рисунок 3 показывает контрастный сигнал CSR и подтверждает наличие существенного разрыва качества между лучшими и худшими объяснениями одной модели.

![Рисунок 3. Контрастный педагогический сигнал CSR](figures/fig3_csr_contrast_ru.png)

**Рисунок 3. Контрастный педагогический сигнал CSR.**
CSR используется как диагностический анализ, а не как основной итоговый метод. Доля построенных контрастных пар составляет 0.5111 при k=3 и 0.5889 при k=5; средний разрыв качества между лучшим и худшим объяснением — около 5.4–5.8 балла. Это показывает, что одна и та же модель может порождать резко различающиеся по педагогическому качеству объяснения для одной задачи.
""".strip(
        "\n"
    )

    pat_72 = r"(Интерпретация\.[^\n]*контроллера риска[^\n]*\n)"
    if re.search(pat_72, text2):
        text2 = re.sub(pat_72, r"\1\n" + fig2_block + "\n", text2, count=1)
    else:
        text2 = text2.replace(
            "### 7.3. Итоговые эффекты DBR на Gen2",
            fig2_block + "\n\n### 7.3. Итоговые эффекты DBR на Gen2",
            1,
        )

    pat_73 = r"(Интерпретация\.[^\n]*смешанные и чувствительные к запуску\.[^\n]*\n)"
    if re.search(pat_73, text2):
        text2 = re.sub(pat_73, r"\1\n" + fig1_block + "\n", text2, count=1)
    else:
        text2 = text2.replace(
            "### 7.4. Краткий вывод по разделу результатов",
            fig1_block + "\n\n### 7.4. Краткий вывод по разделу результатов",
            1,
        )

    pat_8 = r"(Интерпретация\.[^\n]*диагностический, а не основной метод\.[^\n]*\n)"
    if re.search(pat_8, text2):
        text2 = re.sub(pat_8, r"\1\n" + fig3_block + "\n", text2, count=1)
    else:
        text2 = text2.replace(
            "## 9. Проверка надежности и воспроизводимости",
            fig3_block + "\n\n## 9. Проверка надежности и воспроизводимости",
            1,
        )

    text2 = text2.replace("[Рисунок будет вставлен перед подачей]", "")
    (out / "manuscript_ru_prefinal.md").write_text(text2, encoding="utf-8")

    appendix_fig = """# Приложение D. Дополнительные рисунки

## Рисунок A1. Дельты DBR по отдельным запускам

![Рисунок A1. Дельты DBR по отдельным запускам](figures/figA1_dbr_deltas_by_seed_ru.png)

**Рисунок A1. Дельты DBR по отдельным запускам.**
Рисунок показывает чувствительность итоговых эффектов DBR к запуску. Для доли скрытых ошибок отрицательная дельта означает улучшение.

## Рисунок A2. Дополнительная парная проверка Qwen3-235B

![Рисунок A2. Дополнительная парная проверка Qwen3-235B](figures/figA2_qwen_pairwise_ru.png)

**Рисунок A2. Дополнительная парная проверка Qwen3-235B.**
Из 48 выбранных пар успешно оценены 39; DBR выбран 13 раз, обычное рециклирование — 8 раз, ничья — 18 раз. Высокая доля ничьих подчёркивает, что это проверка чувствительности, а не основное доказательство.

## Рисунок A3. Обычное рециклирование: межпоколенческая динамика

![Рисунок A3. Обычное рециклирование: межпоколенческая динамика](figures/figA3_pure_recycling_dynamics_ru.png)

**Рисунок A3. Обычное рециклирование: межпоколенческая динамика.**
Рисунок не демонстрирует сильную монотонную форму коллапса; он показывает ненулевую дефектную нагрузку и нестабильность педагогических метрик.
"""
    (out / "appendix_figures.md").write_text(appendix_fig, encoding="utf-8")

    for s, d in {
        src_claim: out / "appendix_claim_status.md",
        src_repro: out / "appendix_reproducibility.md",
        src_tables: out / "appendix_tables.md",
        src_change: out / "theory_change_log.md",
    }.items():
        if s.exists():
            shutil.copy2(s, d)

    meta = src_metadata.read_text(encoding="utf-8")
    meta = meta.replace(
        "Количество рисунков в основном тексте: 3 (заглушки; заменить финальными графиками перед подачей)",
        "Количество рисунков в основном тексте: 3",
    )
    if "Дополнительные рисунки в приложении" not in meta:
        meta += "\n- Дополнительные рисунки в приложении: 3\n"
    (out / "submission_metadata.md").write_text(meta, encoding="utf-8")

    check_lines = src_checklist.read_text(encoding="utf-8").splitlines()
    out_lines = []
    for ln in check_lines:
        if "Заменить заглушки рисунков на финальные изображения или удалить раздел 7.5." in ln:
            out_lines.append("- [x] Заменить заглушки рисунков на финальные изображения или удалить раздел 7.5.")
        else:
            out_lines.append(ln)
    for item in [
        "- [ ] Визуально проверить качество вставленных рисунков в DOCX.",
        "- [ ] Проверить, что рисунки не разрываются некорректно при переносе страниц.",
        "- [ ] Проверить, что все рисунки имеют подписи.",
        "- [ ] Проверить, что основной текст содержит только 3 основных рисунка, а A1–A3 вынесены в приложение.",
    ]:
        if item not in out_lines:
            out_lines.append(item)
    if missing_figs:
        out_lines.append(f"- [ ] ДОПОЛНИТЕЛЬНО: отсутствуют файлы рисунков: {', '.join(missing_figs)}")
    (out / "submission_checklist.md").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    prefinal_log = """# Prefinal Change Log

## Что сделано
1. В основную рукопись вставлены три финальных русскоязычных рисунка:
   - Рисунок 1. Итоговые эффекты DBR на втором поколении.
   - Рисунок 2. Механизм DBR: покрытие и дефектный профиль.
   - Рисунок 3. Контрастный педагогический сигнал CSR.
2. Старые заглушки рисунков удалены.
3. Дополнительные рисунки A1–A3 вынесены в отдельное приложение.
4. Подписи рисунков добавлены на русском языке.
5. Обновлены metadata и checklist.
6. Старый heatmap DBR mechanism не использован.
7. Численные результаты не изменялись.

## Что осталось перед подачей
- Заполнить ФИО, аффилиацию и email.
- Проверить DOI/URL и стиль библиографии.
- Визуально проверить DOCX.
- Проверить качество рисунков после вставки в Word.
- Уточнить доступность кода и данных.
- Выполнить финальную языковую вычитку.
"""

    if src_docx.exists():
        shutil.copy2(src_docx, out / "manuscript_ru_prefinal.docx")
        prefinal_log += (
            "\n- DOCX создан как prefinal-копия исходной версии; актуальная врезка рисунков гарантирована в Markdown-версии. "
            "Для финальной подачи требуется визуальная синхронизация DOCX по manuscript_ru_prefinal.md.\n"
        )
    else:
        prefinal_log += "\n- DOCX не создан: исходный файл не найден.\n"

    if missing_figs:
        prefinal_log += "\n## Техническое замечание\n- Не найдены некоторые файлы рисунков: " + ", ".join(missing_figs) + "\n"

    (out / "prefinal_change_log.md").write_text(prefinal_log, encoding="utf-8")

    m = (out / "manuscript_ru_prefinal.md").read_text(encoding="utf-8").lower()
    checks = {
        "fig1": "figures/fig1_dbr_gen2_outcome_ru.png" in m,
        "fig2": "figures/fig2_dbr_mechanism_ru.png" in m,
        "fig3": "figures/fig3_csr_contrast_ru.png" in m,
        "no_placeholder": "[рисунок будет вставлен перед подачей]" not in m,
        "no_heatmap": "heatmap" not in m,
        "no_model_sudya": "модель-судья" not in m,
        "no_sanity": "sanity-check" not in m,
        "no_auto_auto": "автоматическая автоматическая" not in m,
    }

    print("OUTPUT:", out)
    print("MISSING_FIGURES:", missing_figs)
    print("CHECKS:", checks)
    png_files = sorted(fig_out.glob("*.png"))
    print("PNG_COUNT:", len(png_files))
    for p in png_files:
        print(f"- {p.name}: {p.stat().st_size}")


if __name__ == "__main__":
    main()
