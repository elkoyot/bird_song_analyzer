package com.birdsong.analyzer.data.repository

import com.birdsong.analyzer.data.local.SpeciesBrief
import com.birdsong.analyzer.data.local.SpeciesCard
import com.birdsong.analyzer.data.local.SpeciesDao
import com.birdsong.analyzer.data.model.TaxonFamilyEntity
import com.birdsong.analyzer.data.model.TaxonOrderEntity
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SpeciesRepository @Inject constructor(
    private val dao: SpeciesDao,
) {
    private val lang: String
        get() = if (Locale.getDefault().language == "ru") "ru" else "en"

    /** Resolve a scientific name from ML model output, handling outdated names. */
    suspend fun resolveScientificName(modelName: String): String? {
        // Try direct match first
        if (dao.getByName(modelName) != null) return modelName
        // Try old_latin synonym
        return dao.resolveOldLatinName(modelName)
    }

    /** Full species card for detail screen. */
    suspend fun getSpeciesCard(scientificName: String): SpeciesCard? =
        dao.getSpeciesCard(scientificName, lang)

    /** Search across local names, scientific names, and synonyms. */
    suspend fun search(query: String, limit: Int = 50): List<SpeciesBrief> {
        if (query.isBlank()) return emptyList()
        val byLocal = dao.searchByLocalName(query, lang, limit)
        if (byLocal.isNotEmpty()) return byLocal
        val bySci = dao.searchByScientificName(query, lang, limit)
        if (bySci.isNotEmpty()) return bySci
        return dao.searchBySynonym(query, lang, limit)
    }

    // ── Taxonomy browsing ──

    suspend fun getOrders(taxonClass: String = "Aves"): List<TaxonOrderEntity> =
        dao.getOrdersByClass(taxonClass)

    suspend fun getFamilies(orderId: Int): List<TaxonFamilyEntity> =
        dao.getFamiliesByOrder(orderId)

    suspend fun getSpeciesByFamily(familyId: Int): List<SpeciesBrief> =
        dao.getSpeciesByFamily(familyId, lang)

    // ── Country filter ──

    suspend fun getSpeciesByCountry(
        countryCode: String,
        limit: Int = 50,
        offset: Int = 0,
    ): List<SpeciesBrief> = dao.getSpeciesByCountry(countryCode, lang, limit, offset)

    suspend fun countSpeciesInCountry(countryCode: String): Int =
        dao.countSpeciesInCountry(countryCode)

    // ── Translation ──

    suspend fun getOrderName(latinName: String): String? =
        dao.getTranslation("order", latinName, lang)

    suspend fun getFamilyName(latinName: String): String? =
        dao.getTranslation("family", latinName, lang)
}
