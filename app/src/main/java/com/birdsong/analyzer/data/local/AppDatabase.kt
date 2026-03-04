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
    version = 3,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun geoDao(): GeoDao
    abstract fun fileAnalysisDao(): FileAnalysisDao
}
