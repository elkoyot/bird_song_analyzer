package com.birdsong.analyzer.ml

import android.content.Context
import android.util.Log

/**
 * Loads BirdNET V3.0 labels from a semicolon-delimited CSV.
 *
 * Format: id;sci_name;com_name;gbif;class;order
 * File may have a UTF-8 BOM which is stripped automatically.
 *
 * Common names are cross-referenced with V2.4 labels (which use localized names).
 * Species not present in V2.4 keep the original English name from the CSV.
 */
object BirdNetV30LabelLoader {

    private const val TAG = "V30LabelLoader"
    private const val LABELS_PATH = "birdnet/v30/labels.csv"

    data class V30Labels(
        /** (scientificName, commonName) pairs, index matches model output order. */
        val labels: List<Pair<String, String>>,
        /** Taxonomic class per label index (e.g. "Aves", "Insecta"). Used for non-bird filtering. */
        val classNames: List<String>,
    )

    /**
     * @param birdnetLabels V2.4 labels (scientificName, localizedCommonName) for cross-referencing
     */
    fun load(context: Context, birdnetLabels: List<Pair<String, String>>): V30Labels {
        // Build scientific → localized common name map from V2.4
        val commonNameMap = HashMap<String, String>(birdnetLabels.size)
        for ((scientific, common) in birdnetLabels) {
            commonNameMap[scientific] = common
        }

        val lines = context.assets.open(LABELS_PATH).bufferedReader().use { reader ->
            reader.lineSequence()
                .drop(1) // skip header
                .filter { it.isNotBlank() }
                .toList()
        }

        val labels = ArrayList<Pair<String, String>>(lines.size)
        val classNames = ArrayList<String>(lines.size)
        var matched = 0

        for (line in lines) {
            // Strip BOM if present on first data line
            val clean = line.trimStart('\uFEFF')
            val parts = clean.split(';')
            require(parts.size >= 6) { "Invalid V3.0 label line: $clean" }
            // id;sci_name;com_name;gbif;class;order
            val sciName = parts[1]
            val csvComName = parts[2]
            val className = parts[4]

            // Use localized name from V2.4 if available, otherwise keep CSV name
            val commonName = commonNameMap[sciName]
            if (commonName != null) matched++
            labels.add(sciName to (commonName ?: csvComName))
            classNames.add(className)
        }

        val avesCount = classNames.count { it == "Aves" }
        Log.i(TAG, "Loaded ${labels.size} V3.0 labels ($avesCount Aves, ${labels.size - avesCount} non-bird), " +
            "$matched/${labels.size} localized via V2.4")

        return V30Labels(labels = labels, classNames = classNames)
    }
}
