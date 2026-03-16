package com.birdsong.analyzer.ml

import android.content.Context
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * Maps model label indices to species reference data.
 *
 * Loaded from a CSV file in assets (e.g. `birdnet/v24/model_map.csv`).
 * Format: `labelIndex,modelLabel,scientificName,taxonClass`
 *
 * Provides O(1) lookup by label index.
 */
class ModelMap private constructor(
    /** Model label → current scientific name in reference DB (null = noise/unknown). */
    private val scientificNames: Array<String?>,
    /** Model label → taxon class ("Aves", "Mammalia", etc., empty = unknown). */
    private val taxonClasses: Array<String>,
) {
    val size: Int get() = scientificNames.size

    /** Resolve label index to current scientific name, or null if not in reference. */
    fun getScientificName(labelIndex: Int): String? =
        scientificNames.getOrNull(labelIndex)

    /** Get taxon class for label index. */
    fun getTaxonClass(labelIndex: Int): String =
        taxonClasses.getOrElse(labelIndex) { "" }

    companion object {
        private const val TAG = "ModelMap"

        /** Load model map from a CSV asset file. */
        fun fromAsset(context: Context, assetPath: String): ModelMap {
            val entries = mutableListOf<Triple<Int, String?, String>>()
            var maxIndex = 0

            context.assets.open(assetPath).use { stream ->
                BufferedReader(InputStreamReader(stream)).use { reader ->
                    // Skip header
                    reader.readLine()
                    reader.forEachLine { line ->
                        val parts = line.split(",", limit = 4)
                        if (parts.size >= 4) {
                            val idx = parts[0].toIntOrNull() ?: return@forEachLine
                            val sciName = parts[2].takeIf { it.isNotBlank() }
                            val taxon = parts[3]
                            entries.add(Triple(idx, sciName, taxon))
                            if (idx > maxIndex) maxIndex = idx
                        }
                    }
                }
            }

            val names = arrayOfNulls<String?>(maxIndex + 1)
            val classes = Array(maxIndex + 1) { "" }

            for ((idx, sciName, taxon) in entries) {
                names[idx] = sciName
                classes[idx] = taxon
            }

            Log.i(TAG, "Loaded $assetPath: ${entries.size} entries, max index $maxIndex")
            return ModelMap(names, classes)
        }
    }
}
