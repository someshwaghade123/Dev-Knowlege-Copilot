// mobile/app/_layout.tsx
// Root layout required by Expo Router — wraps all screens
import { Stack } from "expo-router";

export default function RootLayout() {
    return (
        <Stack
            screenOptions={{
                headerStyle: { backgroundColor: "#0f172a" },
                headerTintColor: "#f8fafc",
                headerTitleStyle: { fontWeight: "bold" },
                contentStyle: { backgroundColor: "#0f172a" },
            }}
        />
    );
}
