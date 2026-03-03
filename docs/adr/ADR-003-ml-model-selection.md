# ADR-003: ML Model Selection — BirdNET

Status: Accepted
Date: 2026-02-19
Deciders: @owner

## Context

Приложению необходима ML-модель для распознавания видов птиц по аудио. Модель должна работать на устройстве (offline), поддерживать европейские виды и конвертироваться в TensorFlow Lite.

Лицензия BirdNET — **CC BY-NC-SA 4.0** (только некоммерческое использование).

## Decision

Используем **BirdNET** (TU Chemnitz) через TensorFlow Lite.

- Источник: https://github.com/kahst/BirdNET-Analyzer
- Формат на устройстве: `.tflite`
- Целевая точность: 80-90%
- Покрытие: европейские виды птиц

**Стратегия по лицензии:** начинаем с BirdNET для MVP. Перед монетизацией — одно из:
1. Получить коммерческую лицензию у TU Chemnitz
2. Обучить собственную модель на собранных данных + открытых датасетах (Xeno-canto)
3. Использовать модели из Bird-CLEF с подходящей лицензией

## Consequences

### Positive
- Готовая, хорошо протестированная модель — быстрый старт без ML-экспертизы
- Широкое покрытие видов, включая европейские
- Активное сообщество и документация
- TFLite версия оптимизирована для мобильных устройств

### Negative
- CC BY-NC-SA 4.0 запрещает коммерческое использование — нужно решить до монетизации
- Зависимость от стороннего проекта (обновления, поддержка)
- Размер модели увеличивает APK (необходимо оценить при интеграции)

### Neutral
- Модель можно обновлять без переустановки приложения (download at first launch или через assets)
- Данные собранные приложением могут использоваться для обучения собственной модели в будущем

## Alternatives Considered

- **Bird-CLEF winners (Kaggle)** — высокая точность, но нужна самостоятельная конвертация в TFLite и оптимизация. Лицензии разные
- **Собственная модель** — полный контроль и нет лицензионных ограничений, но огромный объём работы и не факт, что достигнем качества BirdNET

## Amendment 2026-03-04: Добавление BirdNET V3.0 EUNA

### Контекст

BirdNET V3.0 EUNA (Eurasia + North Africa) — новая модель от TU Chemnitz с улучшенным покрытием целевого региона и коммерчески допустимой лицензией.

### Решение

Добавлен **BirdNET V3.0 EUNA** как второй классификатор (dual-model система):

- **Runtime:** ONNX Runtime Android (`com.microsoft.onnxruntime:onnxruntime-android:1.21.0`)
- **Формат:** FP32 ONNX
- **Параметры:** 32 kHz, 5 секунд (160 000 samples)
- **Классы:** 1225 (972 Aves)
- **Лицензия:** CC BY-SA 4.0 — **коммерческое использование разрешено**
- **Размещение:** не в assets (слишком большой), sideload через `adb push` в `context.filesDir/models/birdnet_v30/birdnet_v30_euna.onnx`
- **Labels:** `assets/birdnet/v30/labels.csv` (`;`-delimited CSV, UTF-8 BOM)
- **Non-bird фильтр:** по taxonomic class (`classNames[i] == "Aves"`)
- **Гео-фильтр:** через V2.4 MetaProfile (маппинг V3.0 → V2.4 label index)

### Реализация

- `BirdNetV30Classifier.kt` — ONNX Runtime inference
- `BirdNetV30LabelLoader.kt` — парсинг labels с cross-reference на V2.4 для локализации
- `ClassifierFactory.kt` — `MODEL_BIRDNET="birdnet_v24"`, `MODEL_BIRDNET_V30="birdnet_v30"`; lazy V3.0 init, проверка доступности `isBirdNetV30Available()`
- `AudioResampler.kt` — 48 kHz → 32 kHz (linear interpolation)

### Влияние на стратегию лицензирования

V3.0 EUNA под CC BY-SA 4.0 снимает лицензионное ограничение для коммерческого использования. Стратегия обновлена:
1. **Для MVP:** V3.0 EUNA как основная модель (CC BY-SA 4.0 — ОК)
2. **V2.4 (CC BY-NC-SA 4.0):** сохраняется для сравнения и гео-фильтрации, но не должна быть единственным классификатором в коммерческой версии

### Удалено

Perch V2 (ранее альтернативный классификатор) — полностью удалён в пользу V3.0 EUNA
