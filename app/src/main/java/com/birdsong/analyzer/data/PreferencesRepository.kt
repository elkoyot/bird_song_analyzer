package com.birdsong.analyzer.data

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import com.birdsong.analyzer.ml.ClassifierFactory
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class PreferencesRepository @Inject constructor(
    private val dataStore: DataStore<Preferences>,
) {
    val countryCode: Flow<String> = dataStore.data.map { it[KEY_COUNTRY] ?: DEFAULT_COUNTRY }

    val regionCode: Flow<String?> = dataStore.data.map { it[KEY_REGION] }

    val activeModel: Flow<String> = dataStore.data.map {
        val stored = it[KEY_MODEL] ?: ClassifierFactory.MODEL_BIRDNET
        // Migrate legacy "perch_v2" preference to default BirdNET
        if (stored == "perch_v2") ClassifierFactory.MODEL_BIRDNET else stored
    }

    suspend fun setCountry(code: String) {
        dataStore.edit { prefs ->
            prefs[KEY_COUNTRY] = code
            prefs.remove(KEY_REGION)  // сброс региона при смене страны
        }
    }

    suspend fun setRegion(code: String?) {
        dataStore.edit { prefs ->
            if (code != null) prefs[KEY_REGION] = code
            else prefs.remove(KEY_REGION)
        }
    }

    suspend fun setActiveModel(model: String) {
        dataStore.edit { prefs -> prefs[KEY_MODEL] = model }
    }

    companion object {
        private val KEY_COUNTRY = stringPreferencesKey("country_code")
        private val KEY_REGION  = stringPreferencesKey("region_code")
        private val KEY_MODEL   = stringPreferencesKey("active_model")
        const val DEFAULT_COUNTRY = "BY"
    }
}
