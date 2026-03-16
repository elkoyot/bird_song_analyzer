package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey

@Entity(
    tableName = "taxonomy_synonym",
    foreignKeys = [
        ForeignKey(
            entity = SpeciesEntity::class,
            parentColumns = ["scientific_name"],
            childColumns = ["scientific_name"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
)
data class TaxonomySynonymEntity(
    @PrimaryKey val synonym: String,
    @ColumnInfo(name = "scientific_name") val scientificName: String,
    val type: String,
)
