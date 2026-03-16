package com.birdsong.analyzer.data.repository

import androidx.room.withTransaction
import com.birdsong.analyzer.data.local.FileAnalysisDao
import com.birdsong.analyzer.data.local.FileAnalysisSummary
import com.birdsong.analyzer.data.local.UserDatabase
import com.birdsong.analyzer.data.model.FileAnalysisEntity
import com.birdsong.analyzer.data.model.FileDetectionEntity
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class FileAnalysisRepository @Inject constructor(
    private val database: UserDatabase,
    private val dao: FileAnalysisDao,
) {

    suspend fun saveAnalysis(
        analysis: FileAnalysisEntity,
        detections: List<FileDetectionEntity>,
    ) {
        database.withTransaction {
            dao.insertAnalysis(analysis)
            dao.insertDetections(detections)
        }
    }

    fun getAllSummaries(): Flow<List<FileAnalysisSummary>> = dao.getAllSummaries()

    fun getAllAnalyses(): Flow<List<FileAnalysisEntity>> = dao.getAllAnalyses()

    suspend fun getAnalysisById(id: String): FileAnalysisEntity? = dao.getAnalysisById(id)

    suspend fun getDetectionsForAnalysis(analysisId: String): List<FileDetectionEntity> =
        dao.getDetectionsForAnalysis(analysisId)

    suspend fun deleteAnalysis(id: String) = dao.deleteAnalysis(id)

    suspend fun count(): Int = dao.count()
}
