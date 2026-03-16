package com.birdsong.analyzer.data.local

import androidx.room.Dao
import androidx.room.Query
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.data.model.MlModelEntity

@Dao
interface GeoDao {

    @Query("SELECT * FROM geo_entity WHERE type = 'continent' ORDER BY sort_order")
    suspend fun getContinents(): List<GeoEntity>

    @Query("SELECT * FROM geo_entity WHERE parent_code = :parentCode ORDER BY name_en")
    suspend fun getChildren(parentCode: String): List<GeoEntity>

    @Query("SELECT * FROM geo_entity WHERE code = :code")
    suspend fun getByCode(code: String): GeoEntity?

    @Query(
        """
        WITH RECURSIVE ancestors(code) AS (
            VALUES(:geoCode)
            UNION ALL
            SELECT g.parent_code FROM geo_entity g
            JOIN ancestors a ON g.code = a.code
            WHERE g.parent_code IS NOT NULL
        )
        SELECT m.* FROM ml_model m
        JOIN geo_model gm ON gm.model_id = m.id
        WHERE gm.geo_code IN (SELECT code FROM ancestors)
        """,
    )
    suspend fun getModelsForGeo(geoCode: String): List<MlModelEntity>

    @Query("SELECT COUNT(*) FROM geo_entity WHERE parent_code = :parentCode")
    suspend fun getChildrenCount(parentCode: String): Int

    @Query("SELECT COUNT(*) FROM geo_entity")
    suspend fun count(): Int
}
