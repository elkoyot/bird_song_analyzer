# ADR-005: Data Storage Strategy — Local Only

Status: Accepted
Date: 2026-02-19
Deciders: @owner

## Context

Бэкенд на текущем этапе не разрабатывается. Необходимо решить, как хранить данные наблюдений на устройстве и стоит ли закладывать абстракции для будущей синхронизации.

## Decision

**Только локальное хранение. Без абстракций для синхронизации.**

- **Метаданные:** Room (SQLite) — наблюдения, распознанные виды, настройки
- **Аудиофайлы:** Internal Storage (`app-specific directory`)
- **Preferences:** DataStore (Proto или Preferences)
- **Нет Remote data source, нет sync manager**

Схема Room (основные entities):

- `ObservationEntity` — id, speciesId, observedAt, latitude, longitude, locationAccuracy, audioFilePath, confidence, recordingMode, deviceInfo, notes
- `SpeciesEntity` — id, scientificName, commonNameEn, commonNameRu, family, isRare
- `SettingsEntity` — через DataStore (язык, тема, настройки записи)

## Consequences

### Positive
- Минимум кода — быстрый старт
- Нет сложности синхронизации, конфликтов, состояний сети
- Room + Clean Architecture repository pattern достаточно гибкий — Remote data source можно добавить позже без переписывания domain/presentation слоёв

### Negative
- Данные привязаны к устройству — потеря при удалении приложения
- Нет бэкапа в облаке
- Ограничение хранилища устройства (нужна стратегия очистки старых записей)

### Neutral
- Repository interface в domain слое абстрагирует Room — при добавлении бэкенда меняется только реализация в data слое
- Миграции Room при изменении схемы нужно планировать заранее

## Alternatives Considered

- **Локально + интерфейсы для синхронизации** — Remote/Local data source pattern сразу. Отклонено: лишний код без реального использования, YAGNI
- **Firebase Realtime DB / Firestore** — облачное хранение из коробки. Отклонено: бэкенд не рассматривается на этом этапе, vendor lock-in
