package com.birdsong.analyzer.ml

import android.content.Context
import org.json.JSONObject

/**
 * Genus → taxonomic family lookup, loaded from `genus_families.json`.
 *
 * Used to suppress model confusion between species of the same family
 * (e.g. Sylvia borin and Curruca nisoria are both Sylviidae).
 */
class FamilyTaxonomy(private val genusToFamily: Map<String, String>) {

    /** Returns family name for a species scientific name, or null if unknown. */
    fun getFamily(scientificName: String): String? {
        val genus = scientificName.substringBefore(' ')
        return genusToFamily[genus]
    }

    /** Returns true if two scientific names belong to the same family. */
    fun sameFamily(a: String, b: String): Boolean {
        val fa = getFamily(a) ?: return false
        val fb = getFamily(b) ?: return false
        return fa == fb
    }

    companion object {
        private const val ASSET_PATH = "birdnet/v24/genus_families.json"

        fun loadFromAssets(context: Context): FamilyTaxonomy {
            val text = context.assets.open(ASSET_PATH).bufferedReader().use { it.readText() }
            val json = JSONObject(text)
            val map = HashMap<String, String>(json.length())
            for (key in json.keys()) {
                map[key] = json.getString(key)
            }
            return FamilyTaxonomy(map)
        }
    }
}
