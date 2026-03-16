package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity

@Entity(
    tableName = "translation",
    primaryKeys = ["entity_type", "entity_key", "lang"],
)
data class TranslationEntity(
    @ColumnInfo(name = "entity_type") val entityType: String,
    @ColumnInfo(name = "entity_key") val entityKey: String,
    val lang: String,
    val name: String,
)
