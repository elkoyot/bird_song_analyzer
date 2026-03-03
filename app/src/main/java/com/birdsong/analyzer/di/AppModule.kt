package com.birdsong.analyzer.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.sqlite.db.SupportSQLiteDatabase
import com.birdsong.analyzer.data.local.GeoDao
import com.birdsong.analyzer.data.local.GeoDatabase
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

    @Provides
    @Singleton
    fun provideDataStore(@ApplicationContext context: Context): DataStore<Preferences> =
        context.userPrefsDataStore

    @Provides
    @Singleton
    fun provideGeoDatabase(@ApplicationContext context: Context): GeoDatabase {
        lateinit var database: GeoDatabase
        database = Room.databaseBuilder(context, GeoDatabase::class.java, "geo.db")
            .fallbackToDestructiveMigration()
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
    fun provideGeoDao(db: GeoDatabase): GeoDao = db.geoDao()
}
