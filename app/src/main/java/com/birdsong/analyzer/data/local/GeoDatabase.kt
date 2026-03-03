package com.birdsong.analyzer.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.data.model.GeoModelEntity
import com.birdsong.analyzer.data.model.MlModelEntity

@Database(
    entities = [GeoEntity::class, MlModelEntity::class, GeoModelEntity::class],
    version = 2,
    exportSchema = false,
)
abstract class GeoDatabase : RoomDatabase() {
    abstract fun geoDao(): GeoDao
}
