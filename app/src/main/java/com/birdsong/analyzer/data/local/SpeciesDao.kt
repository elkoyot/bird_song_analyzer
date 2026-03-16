package com.birdsong.analyzer.data.local

import androidx.room.Dao
import androidx.room.Query
import com.birdsong.analyzer.data.model.SpeciesEntity
import com.birdsong.analyzer.data.model.TaxonFamilyEntity
import com.birdsong.analyzer.data.model.TaxonOrderEntity

/** Lightweight projection for species list / search results. */
data class SpeciesBrief(
    val scientificName: String,
    val nameLocal: String?,
    val nameEn: String?,
    val familyLatin: String?,
    val iucnStatus: String?,
)

/** Full species card with taxonomy chain. */
data class SpeciesCard(
    val scientificName: String,
    val nameLocal: String?,
    val nameEn: String?,
    val genus: String?,
    val familyLatin: String?,
    val familyLocal: String?,
    val orderLatin: String?,
    val orderLocal: String?,
    val taxonClass: String?,
    val iucnStatus: String?,
)

@Dao
interface SpeciesDao {

    // ── Single species lookup ──

    @Query("SELECT * FROM species WHERE scientific_name = :name")
    suspend fun getByName(name: String): SpeciesEntity?

    /** Resolve an outdated Latin name to the current one. */
    @Query("SELECT scientific_name FROM taxonomy_synonym WHERE synonym = :oldName AND type = 'old_latin'")
    suspend fun resolveOldLatinName(oldName: String): String?

    /** Get full species card with taxonomy and localized names. */
    @Query("""
        SELECT s.scientific_name AS scientificName,
               sn_loc.name AS nameLocal,
               sn_en.name AS nameEn,
               s.genus,
               tf.latin_name AS familyLatin,
               t_fam.name AS familyLocal,
               tor.latin_name AS orderLatin,
               t_ord.name AS orderLocal,
               tor.taxon_class AS taxonClass,
               s.iucn_status AS iucnStatus
        FROM species s
        LEFT JOIN species_name sn_loc ON sn_loc.scientific_name = s.scientific_name AND sn_loc.lang = :lang
        LEFT JOIN species_name sn_en ON sn_en.scientific_name = s.scientific_name AND sn_en.lang = 'en'
        LEFT JOIN taxon_family tf ON tf.id = s.family_id
        LEFT JOIN translation t_fam ON t_fam.entity_type = 'family' AND t_fam.entity_key = tf.latin_name AND t_fam.lang = :lang
        LEFT JOIN taxon_order tor ON tor.id = tf.order_id
        LEFT JOIN translation t_ord ON t_ord.entity_type = 'order' AND t_ord.entity_key = tor.latin_name AND t_ord.lang = :lang
        WHERE s.scientific_name = :name
    """)
    suspend fun getSpeciesCard(name: String, lang: String = "ru"): SpeciesCard?

    // ── Search ──

    /** Search by localized name (partial match). */
    @Query("""
        SELECT s.scientific_name AS scientificName,
               sn_loc.name AS nameLocal,
               sn_en.name AS nameEn,
               tf.latin_name AS familyLatin,
               s.iucn_status AS iucnStatus
        FROM species_name sn
        JOIN species s ON s.scientific_name = sn.scientific_name
        LEFT JOIN species_name sn_loc ON sn_loc.scientific_name = s.scientific_name AND sn_loc.lang = :lang
        LEFT JOIN species_name sn_en ON sn_en.scientific_name = s.scientific_name AND sn_en.lang = 'en'
        LEFT JOIN taxon_family tf ON tf.id = s.family_id
        WHERE sn.name LIKE '%' || :query || '%' AND sn.lang = :lang
        ORDER BY sn.name
        LIMIT :limit
    """)
    suspend fun searchByLocalName(query: String, lang: String = "ru", limit: Int = 50): List<SpeciesBrief>

    /** Search by scientific name (partial match). */
    @Query("""
        SELECT s.scientific_name AS scientificName,
               sn_loc.name AS nameLocal,
               sn_en.name AS nameEn,
               tf.latin_name AS familyLatin,
               s.iucn_status AS iucnStatus
        FROM species s
        LEFT JOIN species_name sn_loc ON sn_loc.scientific_name = s.scientific_name AND sn_loc.lang = :lang
        LEFT JOIN species_name sn_en ON sn_en.scientific_name = s.scientific_name AND sn_en.lang = 'en'
        LEFT JOIN taxon_family tf ON tf.id = s.family_id
        WHERE s.scientific_name LIKE '%' || :query || '%'
        ORDER BY s.scientific_name
        LIMIT :limit
    """)
    suspend fun searchByScientificName(query: String, lang: String = "ru", limit: Int = 50): List<SpeciesBrief>

    /** Search by outdated synonym. */
    @Query("""
        SELECT s.scientific_name AS scientificName,
               sn_loc.name AS nameLocal,
               sn_en.name AS nameEn,
               tf.latin_name AS familyLatin,
               s.iucn_status AS iucnStatus
        FROM taxonomy_synonym ts
        JOIN species s ON s.scientific_name = ts.scientific_name
        LEFT JOIN species_name sn_loc ON sn_loc.scientific_name = s.scientific_name AND sn_loc.lang = :lang
        LEFT JOIN species_name sn_en ON sn_en.scientific_name = s.scientific_name AND sn_en.lang = 'en'
        LEFT JOIN taxon_family tf ON tf.id = s.family_id
        WHERE ts.synonym LIKE '%' || :query || '%'
        ORDER BY ts.synonym
        LIMIT :limit
    """)
    suspend fun searchBySynonym(query: String, lang: String = "ru", limit: Int = 50): List<SpeciesBrief>

    // ── Taxonomy browsing ──

    @Query("""
        SELECT * FROM taxon_order
        WHERE taxon_class = :taxonClass
        ORDER BY latin_name
    """)
    suspend fun getOrdersByClass(taxonClass: String = "Aves"): List<TaxonOrderEntity>

    @Query("SELECT * FROM taxon_family WHERE order_id = :orderId ORDER BY latin_name")
    suspend fun getFamiliesByOrder(orderId: Int): List<TaxonFamilyEntity>

    /** Species in a family, with localized name. */
    @Query("""
        SELECT s.scientific_name AS scientificName,
               sn_loc.name AS nameLocal,
               sn_en.name AS nameEn,
               tf.latin_name AS familyLatin,
               s.iucn_status AS iucnStatus
        FROM species s
        LEFT JOIN species_name sn_loc ON sn_loc.scientific_name = s.scientific_name AND sn_loc.lang = :lang
        LEFT JOIN species_name sn_en ON sn_en.scientific_name = s.scientific_name AND sn_en.lang = 'en'
        LEFT JOIN taxon_family tf ON tf.id = s.family_id
        WHERE s.family_id = :familyId
        ORDER BY s.scientific_name
    """)
    suspend fun getSpeciesByFamily(familyId: Int, lang: String = "ru"): List<SpeciesBrief>

    // ── Country filter ──

    /** Species present in a specific country. */
    @Query("""
        SELECT s.scientific_name AS scientificName,
               sn_loc.name AS nameLocal,
               sn_en.name AS nameEn,
               tf.latin_name AS familyLatin,
               s.iucn_status AS iucnStatus
        FROM species_country sc
        JOIN species s ON s.scientific_name = sc.scientific_name
        LEFT JOIN species_name sn_loc ON sn_loc.scientific_name = s.scientific_name AND sn_loc.lang = :lang
        LEFT JOIN species_name sn_en ON sn_en.scientific_name = s.scientific_name AND sn_en.lang = 'en'
        LEFT JOIN taxon_family tf ON tf.id = s.family_id
        WHERE sc.country_code = :countryCode
        ORDER BY sn_loc.name
        LIMIT :limit OFFSET :offset
    """)
    suspend fun getSpeciesByCountry(
        countryCode: String,
        lang: String = "ru",
        limit: Int = 50,
        offset: Int = 0,
    ): List<SpeciesBrief>

    @Query("SELECT COUNT(*) FROM species_country WHERE country_code = :countryCode")
    suspend fun countSpeciesInCountry(countryCode: String): Int

    // ── Translation helper ──

    @Query("SELECT name FROM translation WHERE entity_type = :type AND entity_key = :key AND lang = :lang")
    suspend fun getTranslation(type: String, key: String, lang: String): String?
}
