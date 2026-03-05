package com.birdsong.analyzer.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.birdsong.analyzer.data.model.FileAnalysisEntity
import com.birdsong.analyzer.data.model.FileDetectionEntity
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.data.model.GeoModelEntity
import com.birdsong.analyzer.data.model.MlModelEntity

@Database(
    entities = [
        GeoEntity::class,
        MlModelEntity::class,
        GeoModelEntity::class,
        FileAnalysisEntity::class,
        FileDetectionEntity::class,
    ],
    version = 4,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun geoDao(): GeoDao
    abstract fun fileAnalysisDao(): FileAnalysisDao

    companion object {
        val MIGRATION_3_4 = object : androidx.room.migration.Migration(3, 4) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE file_analysis ADD COLUMN analysis_duration_ms INTEGER NOT NULL DEFAULT 0")
            }
        }
    }
}
