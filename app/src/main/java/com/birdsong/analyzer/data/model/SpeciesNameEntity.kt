package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index

@Entity(
    tableName = "species_name",
    primaryKeys = ["scientific_name", "lang"],
    foreignKeys = [
        ForeignKey(
            entity = SpeciesEntity::class,
            parentColumns = ["scientific_name"],
            childColumns = ["scientific_name"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("name"), Index("lang")],
)
data class SpeciesNameEntity(
    @ColumnInfo(name = "scientific_name") val scientificName: String,
    val lang: String,
    val name: String,
)
