# ADR-007: Testing Strategy — Full From Start

Status: Accepted
Date: 2026-02-19
Deciders: @owner

## Context

Необходимо определить подход к тестированию. Документация требует 60%+ покрытия для бизнес-логики. Проект включает ML-компонент, аудио-обработку и Room — области, где баги сложно отлавливать вручную.

## Decision

**Полное тестирование с самого начала.**

### Стек:

| Уровень | Инструменты | Что тестируем |
|---------|-------------|---------------|
| Unit | JUnit 5 + MockK | UseCase, Repository, ViewModel, ML classifier |
| Flow | Turbine | StateFlow/SharedFlow в ViewModel |
| DB | Room in-memory | DAO queries, migrations |
| UI | Compose UI Testing | Экраны, навигация, user flows |

### Зависимости (testImplementation):

- `junit-jupiter` — JUnit 5
- `mockk` — мокирование (идиоматичный Kotlin)
- `turbine` — тестирование Flow
- `kotlinx-coroutines-test` — тестирование корутин
- `androidx.compose.ui:ui-test-junit4` — Compose UI тесты
- `androidx.room:room-testing` — тестирование миграций Room

### Правила:

- Domain layer (UseCase) — обязательные unit-тесты
- ViewModel — тесты StateFlow через Turbine
- Room DAO — тесты на in-memory базе
- Compose экраны — smoke tests ключевых user flows
- ML classifier — тесты с фиксированными аудио-сэмплами

## Consequences

### Positive
- Высокое покрытие ловит регрессии на ранней стадии
- Тесты Room DAO предотвращают потерю данных при миграциях
- Compose UI тесты проверяют навигацию и взаимодействие

### Negative
- Больше кода на старте — каждый UseCase/ViewModel сопровождается тестами
- Compose UI тесты медленнее unit-тестов
- Тесты ML classifier зависят от наличия .tflite модели в test assets

### Neutral
- CI/CD не настраивается сейчас — тесты запускаются локально. CI добавим позже

## Alternatives Considered

- **Минимальное тестирование** — только unit-тесты для domain. Отклонено: ML и аудио-компоненты слишком рискованно оставлять без тестов, Room миграции — источник data loss
