package com.birdsong.analyzer.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import androidx.room.migration.Migration
import com.birdsong.analyzer.data.local.FileAnalysisDao
import com.birdsong.analyzer.data.local.GeoDao
import com.birdsong.analyzer.data.local.AppDatabase
import com.birdsong.analyzer.data.local.GeoSeedLoader
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import javax.inject.Singleton

private val Context.userPrefsDataStore: DataStore<Preferences> by preferencesDataStore(name = "user_prefs")

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    private val MIGRATION_2_3 = object : Migration(2, 3) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS `file_analysis` (
                    `id` TEXT NOT NULL PRIMARY KEY,
                    `file_name` TEXT NOT NULL,
                    `file_uri` TEXT NOT NULL,
                    `duration_sec` REAL NOT NULL,
                    `file_size_bytes` INTEGER NOT NULL,
                    `region_code` TEXT,
                    `region_label` TEXT,
                    `v30_available` INTEGER NOT NULL,
                    `waveform_data` BLOB,
                    `created_at` INTEGER NOT NULL,
                    `species_count` INTEGER NOT NULL
                )
                """,
            )
            db.execSQL(
                """
                CREATE TABLE IF NOT EXISTS `file_detection` (
                    `id` TEXT NOT NULL PRIMARY KEY,
                    `analysis_id` TEXT NOT NULL,
                    `scientific_name` TEXT NOT NULL,
                    `common_name` TEXT NOT NULL,
                    `start_time_sec` REAL NOT NULL,
                    `end_time_sec` REAL NOT NULL,
                    `v24_confidence` REAL,
                    `v30_confidence` REAL,
                    FOREIGN KEY(`analysis_id`) REFERENCES `file_analysis`(`id`) ON DELETE CASCADE
                )
                """,
            )
            db.execSQL(
                "CREATE INDEX IF NOT EXISTS `index_file_detection_analysis_id` ON `file_detection` (`analysis_id`)",
            )
        }
    }

    @Provides
    @Singleton
    fun provideDataStore(@ApplicationContext context: Context): DataStore<Preferences> =
        context.userPrefsDataStore

    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        lateinit var database: AppDatabase
        database = Room.databaseBuilder(context, AppDatabase::class.java, "geo.db")
            .addMigrations(MIGRATION_2_3)
            .fallbackToDestructiveMigrationFrom(1)
            .addCallback(object : RoomDatabase.Callback() {
                override fun onOpen(db: SupportSQLiteDatabase) {
                    super.onOpen(db)
                    CoroutineScope(Dispatchers.IO).launch {
                        GeoSeedLoader.seed(database.geoDao(), context)
                    }
                }
            })
            .build()
        return database
    }

    @Provides
    fun provideGeoDao(db: AppDatabase): GeoDao = db.geoDao()

    @Provides
    fun provideFileAnalysisDao(db: AppDatabase): FileAnalysisDao = db.fileAnalysisDao()
}
