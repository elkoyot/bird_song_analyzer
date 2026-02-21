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
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    currentLanguage: String = "English",
    currentTheme: String = "System",
    onLanguageClick: () -> Unit = {},
    onThemeClick: () -> Unit = {},
    onPermissionsClick: () -> Unit = {},
    onAboutClick: () -> Unit = {},
) {
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
            title = stringResource(R.string.settings_permissions),
            onClick = onPermissionsClick,
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

@Preview(showBackground = true, showSystemUi = true, name = "Settings")
@Composable
private fun PreviewSettings() {
    BirdSongTheme {
        SettingsScreen()
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Settings - Dark")
@Composable
private fun PreviewSettingsDark() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        SettingsScreen(
            currentLanguage = "Русский",
            currentTheme = "Тёмная",
        )
    }
}
