package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "species",
    foreignKeys = [
        ForeignKey(
            entity = TaxonFamilyEntity::class,
            parentColumns = ["id"],
            childColumns = ["family_id"],
            onDelete = ForeignKey.SET_NULL,
        ),
    ],
    indices = [Index("family_id"), Index("genus")],
)
data class SpeciesEntity(
    @PrimaryKey @ColumnInfo(name = "scientific_name") val scientificName: String,
    @ColumnInfo(name = "family_id") val familyId: Int?,
    val genus: String?,
    @ColumnInfo(name = "iucn_status") val iucnStatus: String?,
)
