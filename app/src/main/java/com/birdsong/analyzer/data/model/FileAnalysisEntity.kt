package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "file_analysis")
data class FileAnalysisEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "file_name") val fileName: String,
    @ColumnInfo(name = "file_uri") val fileUri: String,
    @ColumnInfo(name = "duration_sec") val durationSec: Float,
    @ColumnInfo(name = "file_size_bytes") val fileSizeBytes: Long,
    @ColumnInfo(name = "region_code") val regionCode: String?,
    @ColumnInfo(name = "region_label") val regionLabel: String?,
    @ColumnInfo(name = "v30_available") val v30Available: Boolean,
    @ColumnInfo(name = "waveform_data") val waveformData: ByteArray?,
    @ColumnInfo(name = "created_at") val createdAt: Long,
    @ColumnInfo(name = "species_count") val speciesCount: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FileAnalysisEntity) return false
        return id == other.id
    }

    override fun hashCode(): Int = id.hashCode()
}
