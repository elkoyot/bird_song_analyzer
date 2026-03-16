package com.birdsong.analyzer.di

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import androidx.room.Room
import com.birdsong.analyzer.data.local.FileAnalysisDao
import com.birdsong.analyzer.data.local.GeoDao
import com.birdsong.analyzer.data.local.ReferenceDatabase
import com.birdsong.analyzer.data.local.SpeciesDao
import com.birdsong.analyzer.data.local.UserDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

private val Context.userPrefsDataStore: DataStore<Preferences> by preferencesDataStore(name = "user_prefs")

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideDataStore(@ApplicationContext context: Context): DataStore<Preferences> =
        context.userPrefsDataStore

    /**
     * Pre-built read-only database with geo-data, taxonomy, and species reference.
     * Loaded from assets/db/reference.db via [createFromAsset].
     * On version bump: [fallbackToDestructiveMigration] replaces the DB with the new asset.
     */
    @Provides
    @Singleton
    fun provideReferenceDatabase(@ApplicationContext context: Context): ReferenceDatabase =
        Room.databaseBuilder(context, ReferenceDatabase::class.java, "reference.db")
            .createFromAsset("db/reference.db")
            .fallbackToDestructiveMigration()
            .build()

    /**
     * User data database — analysis results and observations.
     * Persists independently from reference data updates.
     */
    @Provides
    @Singleton
    fun provideUserDatabase(@ApplicationContext context: Context): UserDatabase =
        Room.databaseBuilder(context, UserDatabase::class.java, UserDatabase.DB_NAME)
            .build()

    @Provides
    fun provideGeoDao(db: ReferenceDatabase): GeoDao = db.geoDao()

    @Provides
    fun provideSpeciesDao(db: ReferenceDatabase): SpeciesDao = db.speciesDao()

    @Provides
    fun provideFileAnalysisDao(db: UserDatabase): FileAnalysisDao = db.fileAnalysisDao()
}
