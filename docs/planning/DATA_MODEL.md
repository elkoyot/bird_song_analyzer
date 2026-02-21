# Модель данных Bird Song Analyzer
## Data Schema и Database Design

**Дата:** 2026-02-17

---

## 1. Обзор

Это центральный датасет проекта. Все данные структурированы для максимальной научной ценности и удобства анализа.

### Ключевые принципы:
1. **Нормализация** - избежание дубликатов
2. **Геопривязка** - каждое наблюдение имеет координаты
3. **Временные метки** - точное время и часовой пояс
4. **Качество** - метрики качества записи
5. **Прослеживаемость** - кто, когда, где, как

---

## 2. Основные сущности (Entities)

### 2.1 User (Пользователь)

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE,  -- может быть NULL для анонимных
    password_hash VARCHAR(255),  -- NULL для анонимных
    username VARCHAR(100),
    is_anonymous BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,

    -- Настройки
    settings JSONB DEFAULT '{}',  -- предпочтения пользователя

    -- OAuth интеграции
    ebird_connected BOOLEAN DEFAULT false,
    ebird_user_id VARCHAR(255),
    inaturalist_connected BOOLEAN DEFAULT false,
    inaturalist_user_id VARCHAR(255),

    -- Подписка
    subscription_tier VARCHAR(50) DEFAULT 'free',  -- free, premium, pro
    subscription_expires_at TIMESTAMP WITH TIME ZONE,

    -- Статистика (денормализация для быстрого доступа)
    total_observations INTEGER DEFAULT 0,
    total_species INTEGER DEFAULT 0,

    CONSTRAINT email_required_if_not_anonymous
        CHECK (is_anonymous = true OR email IS NOT NULL)
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);
```

**Поля:**
- `id` - UUID пользователя
- `email` - почта (NULL для анонимных)
- `is_anonymous` - флаг анонимного пользователя
- `settings` - JSON с настройками (язык, уведомления и т.д.)
- `subscription_tier` - уровень подписки
- OAuth поля - для интеграций

---

### 2.2 Species (Виды птиц)

```sql
CREATE TABLE species (
    id SERIAL PRIMARY KEY,
    scientific_name VARCHAR(255) UNIQUE NOT NULL,  -- латинское название
    common_name_en VARCHAR(255) NOT NULL,
    common_name_ru VARCHAR(255),
    common_name_de VARCHAR(255),  -- для будущих локализаций

    -- Классификация
    family VARCHAR(100),
    order_name VARCHAR(100),

    -- Характеристики
    is_rare BOOLEAN DEFAULT false,
    is_migratory BOOLEAN DEFAULT false,
    conservation_status VARCHAR(50),  -- LC, NT, VU, EN, CR

    -- Ареал обитания (упрощенно, в будущем можно GeoJSON)
    habitat_description TEXT,
    regions VARCHAR[] DEFAULT ARRAY[]::VARCHAR[],  -- ['Eastern Europe', 'Western Europe']

    -- Справочная информация
    description TEXT,
    photo_url VARCHAR(500),
    audio_example_url VARCHAR(500),  -- пример голоса из Xeno-canto

    -- Внешние ID для интеграций
    ebird_species_code VARCHAR(10),
    inaturalist_taxon_id INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_species_scientific_name ON species(scientific_name);
CREATE INDEX idx_species_common_name_en ON species(common_name_en);
CREATE INDEX idx_species_is_rare ON species(is_rare);
CREATE INDEX idx_species_regions ON species USING GIN(regions);
```

**Пример записи:**
```json
{
  "id": 42,
  "scientific_name": "Parus major",
  "common_name_en": "Great Tit",
  "common_name_ru": "Большая синица",
  "family": "Paridae",
  "order_name": "Passeriformes",
  "is_rare": false,
  "is_migratory": false,
  "conservation_status": "LC",
  "regions": ["Eastern Europe", "Western Europe", "Northern Europe"],
  "ebird_species_code": "gretit1"
}
```

---

### 2.3 Observation (Наблюдение) - ГЛАВНАЯ ТАБЛИЦА

```sql
CREATE TABLE observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    species_id INTEGER REFERENCES species(id),  -- может быть NULL если не определено

    -- Временные метки
    observed_at TIMESTAMP WITH TIME ZONE NOT NULL,  -- когда наблюдение
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),  -- когда создано в системе
    uploaded_at TIMESTAMP WITH TIME ZONE,  -- когда загружено на сервер

    -- Геолокация
    latitude DECIMAL(10, 8),  -- -90 to 90
    longitude DECIMAL(11, 8),  -- -180 to 180
    altitude DECIMAL(8, 2),  -- метры над уровнем моря (опционально)
    location_accuracy DECIMAL(8, 2),  -- точность GPS в метрах

    -- Распознавание
    recognized_by VARCHAR(50) DEFAULT 'mobile',  -- mobile, server, manual
    mobile_species_id INTEGER,  -- что определило мобильное приложение
    mobile_confidence DECIMAL(5, 2),  -- 0-100%
    server_species_id INTEGER,  -- что определил сервер (может отличаться)
    server_confidence DECIMAL(5, 2),

    -- Аудио
    audio_url VARCHAR(500) NOT NULL,  -- S3 URL
    audio_duration DECIMAL(6, 2),  -- секунды
    audio_format VARCHAR(10) DEFAULT 'mp3',
    audio_size_bytes INTEGER,

    -- Качество записи (вычисляется на сервере)
    quality_score DECIMAL(5, 2),  -- 0-100
    noise_level VARCHAR(20),  -- low, medium, high
    has_multiple_species BOOLEAN DEFAULT false,  -- несколько птиц на записи

    -- Контекст
    recording_mode VARCHAR(50),  -- single, continuous, smart_trap
    device_info JSONB,  -- {device_model, os_version, app_version}

    -- Погода (добавляется на сервере в R3)
    weather_data JSONB,  -- {temp, pressure, wind_speed, precipitation}

    -- Экспедиция (NULL если не в экспедиции)
    expedition_id UUID REFERENCES expeditions(id),

    -- Флаги
    is_verified BOOLEAN DEFAULT false,  -- верифицировано экспертом
    is_public BOOLEAN DEFAULT true,  -- доступно другим пользователям
    is_deleted BOOLEAN DEFAULT false,  -- мягкое удаление

    -- Метаданные
    notes TEXT,  -- заметки пользователя (опционально)
    tags VARCHAR[] DEFAULT ARRAY[]::VARCHAR[],  -- теги

    CONSTRAINT lat_lng_both_or_none
        CHECK ((latitude IS NULL AND longitude IS NULL) OR
               (latitude IS NOT NULL AND longitude IS NOT NULL))
);

-- Индексы для производительности
CREATE INDEX idx_observations_user_id ON observations(user_id);
CREATE INDEX idx_observations_species_id ON observations(species_id);
CREATE INDEX idx_observations_observed_at ON observations(observed_at);
CREATE INDEX idx_observations_location ON observations USING GIST (
    ST_Point(longitude, latitude)
);  -- PostGIS для геопространственных запросов
CREATE INDEX idx_observations_expedition_id ON observations(expedition_id);
CREATE INDEX idx_observations_recording_mode ON observations(recording_mode);
CREATE INDEX idx_observations_is_public ON observations(is_public) WHERE is_public = true;

-- Полнотекстовый поиск (опционально)
CREATE INDEX idx_observations_notes_fulltext ON observations USING GIN(to_tsvector('english', notes));
```

**Пример записи:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "species_id": 42,
  "observed_at": "2026-02-17T06:15:30+03:00",
  "latitude": 53.9045,
  "longitude": 27.5615,
  "location_accuracy": 12.5,
  "mobile_species_id": 42,
  "mobile_confidence": 87.3,
  "server_species_id": 42,
  "server_confidence": 92.1,
  "audio_url": "https://s3.../observations/550e8400.../audio.mp3",
  "audio_duration": 8.5,
  "quality_score": 88.0,
  "noise_level": "low",
  "recording_mode": "smart_trap",
  "device_info": {
    "device_model": "Samsung Galaxy S23",
    "os_version": "Android 14",
    "app_version": "1.0.5"
  },
  "is_public": true
}
```

---

### 2.4 Expedition (Экспедиция) - Релиз 2

```sql
CREATE TABLE expeditions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id),

    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Уникальный код для присоединения
    join_code VARCHAR(10) UNIQUE NOT NULL,

    -- Временные рамки
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,

    -- Статус
    status VARCHAR(20) DEFAULT 'active',  -- active, completed, archived
    is_public BOOLEAN DEFAULT false,

    -- Статистика (денормализация)
    total_participants INTEGER DEFAULT 1,
    total_observations INTEGER DEFAULT 0,
    total_species INTEGER DEFAULT 0,

    -- Настройки
    settings JSONB DEFAULT '{}',  -- {allow_join: true, share_location: true}

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_expeditions_owner_id ON expeditions(owner_id);
CREATE INDEX idx_expeditions_join_code ON expeditions(join_code);
CREATE INDEX idx_expeditions_status ON expeditions(status);
```

---

### 2.5 Expedition_Participant (Участники экспедиции)

```sql
CREATE TABLE expedition_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    expedition_id UUID NOT NULL REFERENCES expeditions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    left_at TIMESTAMP WITH TIME ZONE,

    role VARCHAR(20) DEFAULT 'participant',  -- owner, participant

    -- Статистика участника в экспедиции
    observations_count INTEGER DEFAULT 0,
    species_count INTEGER DEFAULT 0,

    UNIQUE(expedition_id, user_id)
);

CREATE INDEX idx_expedition_participants_expedition_id ON expedition_participants(expedition_id);
CREATE INDEX idx_expedition_participants_user_id ON expedition_participants(user_id);
```

---

### 2.6 Achievement (Достижения) - Релиз 2

```sql
CREATE TABLE achievements (
    id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,  -- first_species, collector_100, rare_bird
    name VARCHAR(255) NOT NULL,
    description TEXT,
    icon_url VARCHAR(500),

    -- Условия получения
    category VARCHAR(50),  -- discovery, quality, activity, region
    tier VARCHAR(20),  -- bronze, silver, gold, platinum
    points INTEGER DEFAULT 0,

    -- Требования (JSON описание логики)
    requirements JSONB NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_achievements_category ON achievements(category);
```

**Пример достижения:**
```json
{
  "id": 1,
  "code": "first_species",
  "name": "Первооткрыватель",
  "description": "Записали свой первый вид птицы",
  "category": "discovery",
  "tier": "bronze",
  "points": 10,
  "requirements": {
    "type": "species_count",
    "threshold": 1
  }
}
```

---

### 2.7 User_Achievement (Полученные достижения)

```sql
CREATE TABLE user_achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    achievement_id INTEGER NOT NULL REFERENCES achievements(id),

    unlocked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Контекст получения
    related_observation_id UUID REFERENCES observations(id),

    UNIQUE(user_id, achievement_id)
);

CREATE INDEX idx_user_achievements_user_id ON user_achievements(user_id);
CREATE INDEX idx_user_achievements_unlocked_at ON user_achievements(unlocked_at);
```

---

### 2.8 Regional_Statistics (Статистика по регионам) - Кэш таблица

```sql
CREATE TABLE regional_statistics (
    id SERIAL PRIMARY KEY,
    region VARCHAR(100) NOT NULL,  -- "Belarus", "Poland", "Germany"
    year INTEGER NOT NULL,
    month INTEGER,  -- NULL = за весь год

    -- Агрегированная статистика
    total_observations INTEGER DEFAULT 0,
    total_species INTEGER DEFAULT 0,
    unique_observers INTEGER DEFAULT 0,

    -- Топ виды (денормализация для быстрого доступа)
    top_species JSONB,  -- [{species_id: 42, count: 150}, ...]

    -- Метаданные
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(region, year, month)
);

CREATE INDEX idx_regional_statistics_region ON regional_statistics(region);
CREATE INDEX idx_regional_statistics_year ON regional_statistics(year);
```

---

## 3. Дополнительные таблицы

### 3.1 Device (Устройства пользователя)

```sql
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    device_token VARCHAR(255) UNIQUE,  -- для push уведомлений
    device_type VARCHAR(20),  -- android, ios (в будущем)
    device_model VARCHAR(100),
    os_version VARCHAR(50),
    app_version VARCHAR(20),

    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_devices_user_id ON devices(user_id);
CREATE INDEX idx_devices_device_token ON devices(device_token);
```

---

### 3.2 Export_Request (Экспорт данных)

```sql
CREATE TABLE export_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),

    format VARCHAR(20) NOT NULL,  -- csv, json, geojson, excel
    filters JSONB,  -- фильтры для экспорта

    status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, completed, failed
    file_url VARCHAR(500),  -- URL готового файла
    expires_at TIMESTAMP WITH TIME ZONE,  -- файл доступен 7 дней

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_export_requests_user_id ON export_requests(user_id);
CREATE INDEX idx_export_requests_status ON export_requests(status);
```

---

### 3.3 Integration_Log (Лог интеграций) - Релиз 3

```sql
CREATE TABLE integration_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    observation_id UUID REFERENCES observations(id),

    integration_type VARCHAR(50) NOT NULL,  -- ebird, inaturalist
    action VARCHAR(50),  -- export, import, sync

    status VARCHAR(20),  -- success, failed
    error_message TEXT,

    request_data JSONB,
    response_data JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_integration_logs_user_id ON integration_logs(user_id);
CREATE INDEX idx_integration_logs_observation_id ON integration_logs(observation_id);
CREATE INDEX idx_integration_logs_created_at ON integration_logs(created_at);
```

---

## 4. Views (Представления)

### 4.1 User Statistics View

```sql
CREATE VIEW user_statistics AS
SELECT
    u.id AS user_id,
    u.username,
    COUNT(DISTINCT o.id) AS total_observations,
    COUNT(DISTINCT o.species_id) AS total_species,
    COUNT(DISTINCT CASE WHEN s.is_rare THEN o.species_id END) AS rare_species_count,
    AVG(o.quality_score) AS avg_quality_score,
    MIN(o.observed_at) AS first_observation_at,
    MAX(o.observed_at) AS last_observation_at,
    COUNT(DISTINCT ua.achievement_id) AS achievements_count
FROM users u
LEFT JOIN observations o ON o.user_id = u.id AND o.is_deleted = false
LEFT JOIN species s ON s.id = o.species_id
LEFT JOIN user_achievements ua ON ua.user_id = u.id
GROUP BY u.id, u.username;
```

---

### 4.2 Species Popularity View

```sql
CREATE VIEW species_popularity AS
SELECT
    s.id AS species_id,
    s.scientific_name,
    s.common_name_en,
    COUNT(o.id) AS observation_count,
    COUNT(DISTINCT o.user_id) AS observer_count,
    AVG(o.quality_score) AS avg_quality_score,
    MAX(o.observed_at) AS last_seen_at
FROM species s
LEFT JOIN observations o ON o.species_id = s.id AND o.is_deleted = false
GROUP BY s.id, s.scientific_name, s.common_name_en
ORDER BY observation_count DESC;
```

---

## 5. Triggers и Функции

### 5.1 Автоматическое обновление статистики пользователя

```sql
CREATE OR REPLACE FUNCTION update_user_statistics()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE users
        SET
            total_observations = total_observations + 1,
            total_species = (
                SELECT COUNT(DISTINCT species_id)
                FROM observations
                WHERE user_id = NEW.user_id AND is_deleted = false
            )
        WHERE id = NEW.user_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE users
        SET
            total_observations = total_observations - 1,
            total_species = (
                SELECT COUNT(DISTINCT species_id)
                FROM observations
                WHERE user_id = OLD.user_id AND is_deleted = false
            )
        WHERE id = OLD.user_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_user_statistics
AFTER INSERT OR DELETE ON observations
FOR EACH ROW
EXECUTE FUNCTION update_user_statistics();
```

---

### 5.2 Проверка достижений при новом наблюдении

```sql
CREATE OR REPLACE FUNCTION check_achievements_on_observation()
RETURNS TRIGGER AS $$
DECLARE
    user_species_count INTEGER;
    user_rare_species_count INTEGER;
BEGIN
    -- Проверяем достижение "Первооткрыватель"
    SELECT total_species INTO user_species_count
    FROM users
    WHERE id = NEW.user_id;

    IF user_species_count = 1 THEN
        INSERT INTO user_achievements (user_id, achievement_id, related_observation_id)
        VALUES (NEW.user_id, (SELECT id FROM achievements WHERE code = 'first_species'), NEW.id)
        ON CONFLICT DO NOTHING;
    END IF;

    -- Проверяем достижение "Коллекционер 10/50/100"
    IF user_species_count = 10 THEN
        INSERT INTO user_achievements (user_id, achievement_id)
        VALUES (NEW.user_id, (SELECT id FROM achievements WHERE code = 'collector_10'))
        ON CONFLICT DO NOTHING;
    END IF;

    -- Можно добавить другие проверки...

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_check_achievements
AFTER INSERT ON observations
FOR EACH ROW
EXECUTE FUNCTION check_achievements_on_observation();
```

---

## 6. Размер датасета (оценка)

### После 1 года работы (10,000 активных пользователей):

**Observations:**
- 10,000 пользователей × 50 записей/год = 500,000 наблюдений
- Размер строки в PostgreSQL: ~500 bytes
- **Итого: ~250 MB** (метаданные)

**Audio files:**
- 500,000 записей × 1 MB (MP3) = **500 GB**

**Total: ~500.25 GB** для первого года

### Рост данных:

| Год | Пользователи | Наблюдения | Метаданные | Аудио | Итого |
|-----|--------------|------------|------------|-------|-------|
| 1   | 10,000       | 500K       | 250 MB     | 500 GB | 500.25 GB |
| 2   | 30,000       | 2M         | 1 GB       | 2 TB   | 2.001 TB |
| 3   | 50,000       | 4M         | 2 GB       | 4 TB   | 4.002 TB |

**Стоимость хранения (AWS S3 Standard):**
- $0.023 per GB/month
- Год 1: 500 GB × $0.023 = $11.5/месяц
- Год 2: 2 TB × $0.023 = $46/месяц

**Оптимизация:**
- S3 Intelligent-Tiering (дешевле для старых данных)
- Удаление дубликатов
- Более агрессивное сжатие (MP3 64kbps для архива)

---

## 7. Запросы и аналитика

### 7.1 Топ наблюдателей региона

```sql
SELECT
    u.username,
    COUNT(DISTINCT o.id) AS observations_count,
    COUNT(DISTINCT o.species_id) AS species_count,
    AVG(o.quality_score) AS avg_quality
FROM users u
JOIN observations o ON o.user_id = u.id
WHERE o.latitude BETWEEN 51.0 AND 56.0  -- пример: Беларусь
  AND o.longitude BETWEEN 23.0 AND 33.0
  AND o.is_deleted = false
  AND o.observed_at >= NOW() - INTERVAL '1 year'
GROUP BY u.id, u.username
ORDER BY observations_count DESC
LIMIT 10;
```

---

### 7.2 Редкие виды в регионе

```sql
SELECT
    s.common_name_en,
    s.scientific_name,
    COUNT(o.id) AS sighting_count,
    MAX(o.observed_at) AS last_seen
FROM species s
JOIN observations o ON o.species_id = s.id
WHERE s.is_rare = true
  AND o.latitude BETWEEN 51.0 AND 56.0
  AND o.longitude BETWEEN 23.0 AND 33.0
  AND o.is_deleted = false
  AND o.observed_at >= NOW() - INTERVAL '1 year'
GROUP BY s.id, s.common_name_en, s.scientific_name
ORDER BY sighting_count DESC;
```

---

### 7.3 Активность по времени суток

```sql
SELECT
    EXTRACT(HOUR FROM o.observed_at AT TIME ZONE 'UTC') AS hour_of_day,
    COUNT(*) AS observation_count,
    COUNT(DISTINCT o.species_id) AS species_diversity
FROM observations o
WHERE o.is_deleted = false
  AND o.observed_at >= NOW() - INTERVAL '3 months'
GROUP BY hour_of_day
ORDER BY hour_of_day;
```

---

## 8. Миграции и версионирование

### Инструменты:
- **Alembic** (Python) - для управления миграциями БД
- **Flyway** (Java) - альтернатива
- **Liquibase** - еще одна альтернатива

### Стратегия миграций:
1. Каждая миграция - отдельный файл
2. Нумерация: `001_initial_schema.sql`, `002_add_expeditions.sql`
3. Откат (downgrade) всегда должен быть возможен
4. Тестирование миграций на staging перед продакшном
5. Бэкап перед каждой миграцией

---

## 9. Безопасность данных

### Персональные данные (GDPR compliance):

**Что считается персональными данными:**
- Email, username
- GPS координаты наблюдений
- IP адреса (в логах)
- Device info

**Меры защиты:**
1. **Шифрование:** Все персональные данные шифруются at-rest и in-transit
2. **Анонимизация:** Возможность использовать приложение анонимно
3. **Право на забвение:** Пользователь может удалить свой аккаунт и все данные
4. **Экспорт данных:** Пользователь может экспортировать все свои данные
5. **Согласие:** Явное согласие на обработку при регистрации

### Публичные vs Приватные данные:

**Публичные (по умолчанию):**
- Наблюдения (species, location, time) - без привязки к user
- Агрегированная статистика

**Приватные:**
- Личные данные пользователя
- Связь наблюдения → пользователь (опционально можно скрыть)
- Точные координаты (можно fuzzing - размытие до ~1км)

---

## Заключение

Эта модель данных:
✅ Нормализована для избежания дубликатов
✅ Оптимизирована для аналитики и машинного обучения
✅ Масштабируема (индексы, партиционирование в будущем)
✅ GDPR compliant
✅ Готова к монетизации (API, датасеты)

**Ключевая ценность:** Структурированный, геопривязанный, временной датасет акустических наблюдений птиц Европы с высоким quality score.
