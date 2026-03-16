package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index

@Entity(
    tableName = "species_country",
    primaryKeys = ["scientific_name", "country_code"],
    foreignKeys = [
        ForeignKey(
            entity = SpeciesEntity::class,
            parentColumns = ["scientific_name"],
            childColumns = ["scientific_name"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("country_code")],
)
data class SpeciesCountryEntity(
    @ColumnInfo(name = "scientific_name") val scientificName: String,
    @ColumnInfo(name = "country_code") val countryCode: String,
)
