package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "ml_model")
data class MlModelEntity(
    @PrimaryKey val id: String,
    val name: String,
    val runtime: String,
    @ColumnInfo(name = "audio_path") val audioPath: String,
    @ColumnInfo(name = "meta_model_path") val metaModelPath: String?,
    @ColumnInfo(name = "labels_path") val labelsPath: String,
    @ColumnInfo(name = "sample_rate") val sampleRate: Int,
    @ColumnInfo(name = "chunk_seconds") val chunkSeconds: Int,
    @ColumnInfo(name = "species_count") val speciesCount: Int,
    @ColumnInfo(name = "is_bundled") val isBundled: Boolean,
)
