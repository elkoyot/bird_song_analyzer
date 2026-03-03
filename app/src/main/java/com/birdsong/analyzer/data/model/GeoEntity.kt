package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.util.Locale

@Entity(
    tableName = "geo_entity",
    foreignKeys = [
        ForeignKey(
            entity = GeoEntity::class,
            parentColumns = ["code"],
            childColumns = ["parent_code"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("parent_code"), Index("type")],
)
data class GeoEntity(
    @PrimaryKey val code: String,
    val type: String,
    @ColumnInfo(name = "parent_code") val parentCode: String?,
    @ColumnInfo(name = "name_ru") val nameRu: String,
    @ColumnInfo(name = "name_en") val nameEn: String,
    @ColumnInfo(name = "min_lat") val minLat: Float?,
    @ColumnInfo(name = "max_lat") val maxLat: Float?,
    @ColumnInfo(name = "min_lon") val minLon: Float?,
    @ColumnInfo(name = "max_lon") val maxLon: Float?,
    @ColumnInfo(name = "buffer_deg") val bufferDeg: Float = 2.5f,
    @ColumnInfo(name = "sort_order") val sortOrder: Int = 0,
) {
    fun displayName(): String =
        if (Locale.getDefault().language == "ru") nameRu else nameEn
}
