package com.birdsong.analyzer.data.local

import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Transaction
import com.birdsong.analyzer.data.model.FileAnalysisEntity
import com.birdsong.analyzer.data.model.FileDetectionEntity
import kotlinx.coroutines.flow.Flow

data class FileAnalysisSummary(
    val id: String,
    @ColumnInfo(name = "file_name") val fileName: String,
    @ColumnInfo(name = "duration_sec") val durationSec: Float,
    @ColumnInfo(name = "file_size_bytes") val fileSizeBytes: Long,
    @ColumnInfo(name = "region_label") val regionLabel: String?,
    @ColumnInfo(name = "created_at") val createdAt: Long,
    @ColumnInfo(name = "species_count") val speciesCount: Int,
    @ColumnInfo(name = "analysis_duration_ms") val analysisDurationMs: Long,
    val waveformSize: Int,
    val detectionCount: Int,
)

@Dao
interface FileAnalysisDao {

    @Insert
    suspend fun insertAnalysis(analysis: FileAnalysisEntity)

    @Insert
    suspend fun insertDetections(detections: List<FileDetectionEntity>)

    @Query("""
        SELECT a.id, a.file_name, a.duration_sec, a.file_size_bytes, a.region_label,
               a.created_at, a.species_count, a.analysis_duration_ms,
               COALESCE(LENGTH(a.waveform_data), 0) AS waveformSize,
               (SELECT COUNT(*) FROM file_detection d WHERE d.analysis_id = a.id) AS detectionCount
        FROM file_analysis a
        ORDER BY a.created_at DESC
    """)
    fun getAllSummaries(): Flow<List<FileAnalysisSummary>>

    @Query("SELECT * FROM file_analysis ORDER BY created_at DESC")
    fun getAllAnalyses(): Flow<List<FileAnalysisEntity>>

    @Transaction
    @Query("SELECT * FROM file_analysis WHERE id = :id")
    suspend fun getAnalysisById(id: String): FileAnalysisEntity?

    @Transaction
    @Query("SELECT * FROM file_detection WHERE analysis_id = :analysisId ORDER BY start_time_sec")
    suspend fun getDetectionsForAnalysis(analysisId: String): List<FileDetectionEntity>

    @Query("DELETE FROM file_analysis WHERE id = :id")
    suspend fun deleteAnalysis(id: String)

    @Query("SELECT COUNT(*) FROM file_analysis")
    suspend fun count(): Int
}
