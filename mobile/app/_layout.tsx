// mobile/app/_layout.tsx
// Root layout required by Expo Router — wraps all screens
import { Stack } from "expo-router";

export default function RootLayout() {
    return (
        <Stack
            screenOptions={{
                headerShown: false,
                contentStyle: { backgroundColor: "#0f172a" },
            }}
        />
    );
}
