# ADR-002: Application Architecture

Status: Accepted
Date: 2026-02-19
Deciders: @owner

## Context

Необходимо определить архитектурный паттерн и модульную структуру Android-приложения. Проект ведётся одним разработчиком, но должен быть структурирован для роста и поддержки.

## Decision

**Паттерн:** MVVM + Clean Architecture
**Модульность:** Один Gradle-модуль `:app` с разделением через пакеты

Структура пакетов:

```
app/src/main/java/com/birdsong/analyzer/
├── data/               # Слой данных
│   ├── local/         # Room DAO, DataStore
│   ├── model/         # Data entities
│   └── repository/    # Repository implementations
├── domain/            # Бизнес-логика
│   ├── model/        # Domain models
│   ├── usecase/      # Use cases
│   └── repository/   # Repository interfaces
├── presentation/      # UI слой
│   ├── home/         # Главный экран
│   ├── result/       # Результат распознавания
│   ├── history/      # История наблюдений
│   ├── settings/     # Настройки
│   ├── theme/        # Material 3 тема
│   └── common/       # Общие UI компоненты
├── ml/               # Машинное обучение
│   ├── classifier/   # BirdNET классификатор
│   └── audio/        # Обработка аудио для ML
├── service/          # Android Services
│   └── recording/    # Foreground Service для записи
└── di/               # Hilt модули
```

Поток данных: `UI (Compose) → ViewModel → UseCase → Repository → Room/Storage`

## Consequences

### Positive
- Clean Architecture обеспечивает тестируемость: domain не зависит от Android
- Один модуль — простая настройка Gradle, быстрый старт
- Пакетная структура обеспечивает логическое разделение без overhead мультимодульности
- Repository pattern абстрагирует источники данных — легко добавить Remote позже

### Negative
- Один модуль не обеспечивает compile-time изоляцию слоёв (domain может случайно импортировать data)
- При росте проекта сборка будет медленнее, чем мультимодульная

### Neutral
- Миграция на мультимодульность возможна позже без переписывания логики — только перемещение пакетов в модули

## Alternatives Considered

- **Мультимодульный проект** (:core, :data, :domain, :feature:*) — строгая изоляция, параллельная сборка. Отклонено: слишком много overhead для solo-разработки на старте
- **MVI** — однонаправленный поток данных. Отклонено: MVVM + StateFlow даёт аналогичный результат с меньшим boilerplate
