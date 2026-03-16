package com.birdsong.analyzer.data.model

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "taxon_order")
data class TaxonOrderEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    @ColumnInfo(name = "latin_name") val latinName: String,
    @ColumnInfo(name = "taxon_class") val taxonClass: String,
)
