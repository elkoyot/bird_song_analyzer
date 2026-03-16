package com.birdsong.analyzer.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.data.model.GeoModelEntity
import com.birdsong.analyzer.data.model.MlModelEntity
import com.birdsong.analyzer.data.model.SpeciesCountryEntity
import com.birdsong.analyzer.data.model.SpeciesEntity
import com.birdsong.analyzer.data.model.SpeciesNameEntity
import com.birdsong.analyzer.data.model.TaxonFamilyEntity
import com.birdsong.analyzer.data.model.TaxonOrderEntity
import com.birdsong.analyzer.data.model.TaxonomySynonymEntity
import com.birdsong.analyzer.data.model.TranslationEntity

/**
 * Read-only reference database loaded from a pre-built asset file.
 * Contains geo-data, taxonomy, species reference, and ML model metadata.
 *
 * Created via [Room.databaseBuilder] with [createFromAsset("db/reference.db")].
 * Updated by shipping a new asset file with a version bump.
 */
@Database(
    entities = [
        // Geo
        GeoEntity::class,
        MlModelEntity::class,
        GeoModelEntity::class,
        // Taxonomy
        TaxonOrderEntity::class,
        TaxonFamilyEntity::class,
        // Species reference
        SpeciesEntity::class,
        SpeciesNameEntity::class,
        TaxonomySynonymEntity::class,
        SpeciesCountryEntity::class,
        // Translations
        TranslationEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
abstract class ReferenceDatabase : RoomDatabase() {
    abstract fun geoDao(): GeoDao
    abstract fun speciesDao(): SpeciesDao
}
