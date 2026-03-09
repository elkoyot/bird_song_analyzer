package com.birdsong.analyzer.presentation.splash

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R

@Composable
fun PermissionScreen(
    denied: Boolean,
    onRequestPermission: () -> Unit,
    onSkip: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(SplashColors.Bg),
        contentAlignment = Alignment.Center,
    ) {
        if (denied) {
            DeniedContent(onGrant = onRequestPermission)
        } else {
            RequestContent(
                onGrant = onRequestPermission,
                onSkip = onSkip,
            )
        }
    }
}

@Composable
private fun RequestContent(
    onGrant: () -> Unit,
    onSkip: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier
                .size(80.dp)
                .background(
                    color = SplashColors.Accent.copy(alpha = 0.2f),
                    shape = CircleShape,
                ),
            contentAlignment = Alignment.Center,
        ) {
            Text(text = "\uD83C\uDF99\uFE0F", fontSize = 36.sp)
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = stringResource(R.string.permission_mic_title),
            color = SplashColors.TextPrimary,
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
        )

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = stringResource(R.string.permission_mic_body),
            color = SplashColors.TextSecondary,
            fontSize = 14.sp,
            textAlign = TextAlign.Center,
            lineHeight = 22.sp,
        )

        Spacer(modifier = Modifier.height(36.dp))

        Button(
            onClick = onGrant,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = SplashColors.Accent),
            shape = RoundedCornerShape(14.dp),
        ) {
            Text(
                text = stringResource(R.string.permission_grant),
                color = Color.Black,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 4.dp),
            )
        }

        Spacer(modifier = Modifier.height(4.dp))

        TextButton(onClick = onSkip) {
            Text(
                text = stringResource(R.string.permission_skip),
                color = SplashColors.TextMuted,
                fontSize = 13.sp,
            )
        }
    }
}

@Composable
private fun DeniedContent(onGrant: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier
                .size(80.dp)
                .background(
                    color = SplashColors.ErrorDim.copy(alpha = 0.094f),
                    shape = CircleShape,
                ),
            contentAlignment = Alignment.Center,
        ) {
            Text(text = "\uD83D\uDEAB", fontSize = 36.sp)
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = stringResource(R.string.permission_denied_title),
            color = SplashColors.TextPrimary,
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
        )

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = stringResource(R.string.permission_denied_body),
            color = SplashColors.TextSecondary,
            fontSize = 14.sp,
            textAlign = TextAlign.Center,
            lineHeight = 22.sp,
        )

        Spacer(modifier = Modifier.height(36.dp))

        Button(
            onClick = onGrant,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = SplashColors.Accent),
            shape = RoundedCornerShape(14.dp),
        ) {
            Text(
                text = stringResource(R.string.permission_grant),
                color = Color.Black,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 4.dp),
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = stringResource(R.string.permission_settings_hint),
            color = SplashColors.TextMuted,
            fontSize = 12.sp,
            textAlign = TextAlign.Center,
        )
    }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
private fun PermissionRequestPreview() {
    PermissionScreen(
        denied = false,
        onRequestPermission = {},
        onSkip = {},
    )
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
private fun PermissionDeniedPreview() {
    PermissionScreen(
        denied = true,
        onRequestPermission = {},
        onSkip = {},
    )
}
