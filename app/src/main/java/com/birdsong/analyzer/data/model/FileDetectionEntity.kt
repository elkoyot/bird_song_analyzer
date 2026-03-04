package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "file_detection",
    foreignKeys = [
        ForeignKey(
            entity = FileAnalysisEntity::class,
            parentColumns = ["id"],
            childColumns = ["analysis_id"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("analysis_id")],
)
data class FileDetectionEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "analysis_id") val analysisId: String,
    @ColumnInfo(name = "scientific_name") val scientificName: String,
    @ColumnInfo(name = "common_name") val commonName: String,
    @ColumnInfo(name = "start_time_sec") val startTimeSec: Float,
    @ColumnInfo(name = "end_time_sec") val endTimeSec: Float,
    @ColumnInfo(name = "v24_confidence") val v24Confidence: Float?,
    @ColumnInfo(name = "v30_confidence") val v30Confidence: Float?,
)
