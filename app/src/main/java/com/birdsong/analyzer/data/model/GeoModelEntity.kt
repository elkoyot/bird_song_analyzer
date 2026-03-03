package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index

@Entity(
    tableName = "geo_model",
    primaryKeys = ["geo_code", "model_id"],
    foreignKeys = [
        ForeignKey(
            entity = GeoEntity::class,
            parentColumns = ["code"],
            childColumns = ["geo_code"],
            onDelete = ForeignKey.CASCADE,
        ),
        ForeignKey(
            entity = MlModelEntity::class,
            parentColumns = ["id"],
            childColumns = ["model_id"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("model_id")],
)
data class GeoModelEntity(
    @ColumnInfo(name = "geo_code") val geoCode: String,
    @ColumnInfo(name = "model_id") val modelId: String,
    @ColumnInfo(name = "is_default") val isDefault: Boolean = false,
)
