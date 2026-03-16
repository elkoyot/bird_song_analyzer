package com.birdsong.analyzer.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.birdsong.analyzer.data.model.FileAnalysisEntity
import com.birdsong.analyzer.data.model.FileDetectionEntity

/**
 * User data database — stores analysis results and observations.
 * This database persists across reference data updates.
 */
@Database(
    entities = [
        FileAnalysisEntity::class,
        FileDetectionEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
abstract class UserDatabase : RoomDatabase() {
    abstract fun fileAnalysisDao(): FileAnalysisDao

    companion object {
        const val DB_NAME = "user.db"
    }
}
