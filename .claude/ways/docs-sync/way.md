---
match: regex
pattern: обнови.*документ|синхрониз.*документ|актуализ.*документ|sync.*doc|update.*doc
commands: git\ commit
scope: agent, subagent
---
# Documentation Sync Way

Перед коммитом или по запросу — проверь, что документация отражает текущее состояние кода.

## Что проверять

### 1. ADR (docs/adr/)

| ADR | Покрывает | Ключевые параметры для проверки |
|-----|-----------|--------------------------------|
| ADR-003 | ML Model Selection | модели, лицензии, форматы, классы |
| ADR-004 | Audio Recording Format | sample rate, codec, битрейт |
| ADR-005 | Data Storage Strategy | Room entities, лимиты хранения |
| ADR-006 | MVP Scope | список экранов, фичи, границы MVP |
| ADR-010 | Live Detection Screen | UI элементы, потоки данных, контролы |

**Действие:** Если код изменил решение, описанное в ADR — обнови ADR (добавь секцию `## Amendment YYYY-MM-DD` с описанием изменения и причиной). Не переписывай оригинальный текст.

### 2. ML Pipeline (docs/planning/AUDIO_ANALYSIS_PIPELINE.md)

Главный технический документ. Проверь соответствие:

- **Пороги и параметры** — inference threshold, anchor threshold, bandpass cutoffs, meta alpha, chunk size, overlap, window size
- **Preprocessing stages** — порядок и логика в AudioChunkProcessor (silence → clipping → spectral → bandpass → post-silence → normalize)
- **Classifier pipeline** — входные форматы, sigmoid, top-K, meta-model blending
- **DetectionAggregator** — window size, confirmation count, non-bird filter
- **2-Path filtering** — anchor vs aggregator-confirmed, family dedup
- **Примеры работы алгоритма** — если добавлена новая логика или изменены пороги, добавь/обнови пример с конкретными числами

### 3. CLAUDE.md (корень проекта)

Проверь секции:
- **ML Model** — актуальные модели, pipeline описание, benchmark результаты
- **Architecture** — структура пакетов, стек технологий
- **MVP Scope** — экраны, фичи

### 4. MEMORY.md (auto-memory)

Проверь, что memory отражает:
- Актуальные модели и их параметры
- Ключевые файлы
- Состояние проекта (что сделано, что нет)

## Как синхронизировать

1. **Прочитай изменённые файлы** (`git diff --cached` или `git diff HEAD`)
2. **Определи затронутые области** — ML, audio, UI, data, DI
3. **Прочитай соответствующие docs** из таблицы выше
4. **Сравни** код с документацией — найди расхождения
5. **Обнови** документы, сохраняя стиль и структуру оригинала
6. **Добавь примеры** для новой/изменённой логики с конкретными числами:

```
Пример: chunk RMS = 0.05, spectral bands = [0.12, 0.45, 0.38, 0.05]
→ lowRatio = 12% (< 95%) → pass
→ highRatio = 5% (< 95%) → pass
→ bandpass 80Hz–15kHz → peak normalize to 0.9
→ BirdNET inference → top-1: Parus major (0.82) → anchor path
```

## Чего НЕ делать

- Не обновляй docs для косметических изменений (форматирование, комментарии)
- Не переписывай ADR — только amendment секции
- Не добавляй документацию для незавершённого кода
- Не трогай docs/benchmarks/ — они обновляются отдельно при запуске бенчмарков
