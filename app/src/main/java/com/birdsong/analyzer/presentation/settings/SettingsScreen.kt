package com.birdsong.analyzer.presentation.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.ConfidenceHigh

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    currentLanguage: String = "English",
    currentTheme: String = "System",
    audioPermissionGranted: Boolean = false,
    locationPermissionGranted: Boolean = false,
    locationLabel: String = "\u2014",
    activeModel: String = "birdnet_v24",
    isV30Available: Boolean = false,
    onLanguageClick: () -> Unit = {},
    onThemeClick: () -> Unit = {},
    onRequestAudioPermission: () -> Unit = {},
    onRequestLocationPermission: () -> Unit = {},
    onAboutClick: () -> Unit = {},
    onLocationClick: () -> Unit = {},
    onModelSelected: (String) -> Unit = {},
) {
    var showModelDialog by remember { mutableStateOf(false) }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.settings_title)) },
        )

        SettingsItem(
            title = stringResource(R.string.settings_language),
            subtitle = currentLanguage,
            onClick = onLanguageClick,
        )
        HorizontalDivider()

        SettingsItem(
            title = stringResource(R.string.settings_theme),
            subtitle = currentTheme,
            onClick = onThemeClick,
        )
        HorizontalDivider()

        SettingsItem(
            title = stringResource(R.string.settings_country_region),
            subtitle = locationLabel,
            onClick = onLocationClick,
        )
        HorizontalDivider()

        val modelLabel = when (activeModel) {
            "birdnet_v30" -> stringResource(R.string.settings_model_v30)
            else -> stringResource(R.string.settings_model_birdnet)
        }
        SettingsItem(
            title = stringResource(R.string.settings_model),
            subtitle = modelLabel,
            onClick = { showModelDialog = true },
        )
        HorizontalDivider()

        Text(
            text = stringResource(R.string.settings_permissions),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(start = 16.dp, top = 16.dp, bottom = 4.dp),
        )

        PermissionItem(
            title = stringResource(R.string.settings_permission_microphone),
            granted = audioPermissionGranted,
            onClick = onRequestAudioPermission,
        )
        HorizontalDivider()

        PermissionItem(
            title = stringResource(R.string.settings_permission_location),
            hint = stringResource(R.string.settings_permission_location_hint),
            granted = locationPermissionGranted,
            onClick = onRequestLocationPermission,
        )
        HorizontalDivider()

        SettingsItem(
            title = stringResource(R.string.settings_about),
            onClick = onAboutClick,
        )
        HorizontalDivider()

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "v0.1.0",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 16.dp),
        )
    }

    // Model picker dialog
    if (showModelDialog) {
        val models = listOf(
            "birdnet_v24" to stringResource(R.string.settings_model_birdnet),
            "birdnet_v30" to stringResource(R.string.settings_model_v30),
        )
        AlertDialog(
            onDismissRequest = { showModelDialog = false },
            title = { Text(stringResource(R.string.settings_select_model)) },
            text = {
                Column {
                    models.forEach { (id, label) ->
                        val isV30 = id == "birdnet_v30"
                        val enabled = !isV30 || isV30Available
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .then(
                                    if (enabled) Modifier.clickable {
                                        showModelDialog = false
                                        onModelSelected(id)
                                    } else Modifier
                                )
                                .padding(vertical = 12.dp, horizontal = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = label,
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = if (enabled) MaterialTheme.colorScheme.onSurface
                                            else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                                )
                                if (isV30 && !isV30Available) {
                                    Text(
                                        text = stringResource(R.string.settings_model_not_available),
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.error,
                                    )
                                }
                            }
                            if (activeModel == id) {
                                Icon(
                                    Icons.Default.Check,
                                    contentDescription = null,
                                    tint = ConfidenceHigh,
                                )
                            }
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showModelDialog = false }) {
                    Text(stringResource(android.R.string.cancel))
                }
            },
        )
    }
}

@Composable
private fun PermissionItem(
    title: String,
    granted: Boolean,
    onClick: () -> Unit,
    hint: String? = null,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .then(if (!granted) Modifier.clickable(onClick = onClick) else Modifier)
            .padding(horizontal = 16.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyLarge,
            )
            if (hint != null && !granted) {
                Text(
                    text = hint,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(
                text = if (granted) stringResource(R.string.settings_permission_granted)
                       else stringResource(R.string.settings_permission_not_granted),
                style = MaterialTheme.typography.bodyMedium,
                color = if (granted) ConfidenceHigh
                        else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (granted) {
            Icon(
                Icons.Default.Check,
                contentDescription = null,
                tint = ConfidenceHigh,
            )
        } else {
            Icon(
                Icons.AutoMirrored.Filled.KeyboardArrowRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun SettingsItem(
    title: String,
    subtitle: String? = null,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyLarge,
            )
            if (subtitle != null) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Icon(
            Icons.AutoMirrored.Filled.KeyboardArrowRight,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// --- Previews ---

@Preview(showBackground = true, showSystemUi = true, name = "Settings — no permissions")
@Composable
private fun PreviewSettings() {
    BirdSongTheme {
        SettingsScreen(
            locationLabel = "Беларусь",
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Settings — all granted")
@Composable
private fun PreviewSettingsGranted() {
    BirdSongTheme {
        SettingsScreen(
            audioPermissionGranted = true,
            locationPermissionGranted = true,
            locationLabel = "Беларусь",
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Settings — Dark")
@Composable
private fun PreviewSettingsDark() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        SettingsScreen(
            currentLanguage = "Русский",
            currentTheme = "Тёмная",
            audioPermissionGranted = true,
            locationPermissionGranted = false,
            locationLabel = "Россия \u00b7 Центральный",
        )
    }
}
