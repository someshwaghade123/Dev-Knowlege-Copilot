// mobile/app/search.tsx
// ─────────────────────────────────────────────────────────────────────────────
// The main screen of the mobile app.
// User types a query, presses Search, sees the answer + citations.

import React, { useState } from "react";
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    StyleSheet,
    SafeAreaView,
    Alert,
    Linking,
} from "react-native";
import { queryDocuments, QueryResponse, Citation } from "../services/api";

// ── Confidence badge colors ────────────────────────────────────────────────────
const CONFIDENCE_COLORS: Record<string, string> = {
    high: "#22c55e",
    medium: "#f59e0b",
    low: "#ef4444",
};

// ── Sub-components ────────────────────────────────────────────────────────────

const CitationCard: React.FC<{ citation: Citation; index: number }> = ({
    citation,
    index,
}) => {
    const handleOpenLink = () => {
        if (citation.source_url) {
            Linking.openURL(citation.source_url).catch(() =>
                Alert.alert("Error", "Could not open link.")
            );
        }
    };

    return (
        <TouchableOpacity
            style={styles.citationCard}
            onPress={handleOpenLink}
            disabled={!citation.source_url}
            activeOpacity={0.7}
        >
            <View style={styles.citationHeader}>
                <Text style={styles.citationNumber}>SOURCE {index + 1}</Text>
                {citation.source_url && <Text style={styles.linkIcon}>🔗</Text>}
            </View>
            <Text style={styles.citationTitle} numberOfLines={1}>{citation.title}</Text>
            <Text style={styles.citationPreview} numberOfLines={4}>
                {citation.text_preview}
            </Text>
        </TouchableOpacity>
    );
};

function ConfidenceBadge({ confidence }: { confidence: string }) {
    return (
        <View style={[styles.badge, { backgroundColor: CONFIDENCE_COLORS[confidence] ?? "#6b7280" }]}>
            <Text style={styles.badgeText}>{confidence.toUpperCase()} CONFIDENCE</Text>
        </View>
    );
}

// ── Main Search Screen ────────────────────────────────────────────────────────

export default function SearchScreen() {
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<QueryResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSearch = async () => {
        const trimmed = query.trim();
        if (!trimmed) {
            Alert.alert("Empty query", "Please enter a question first.");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await queryDocuments(trimmed);
            setResult(response);
        } catch (err: any) {
            setError(err.message ?? "Something went wrong. Is the backend running?");
        } finally {
            setLoading(false);
        }
    };

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">

                {/* Header */}
                <Text style={styles.header}>🧠 Dev Knowledge Copilot</Text>
                <Text style={styles.subtitle}>Ask anything about your technical docs</Text>

                {/* Search input */}
                <View style={styles.inputRow}>
                    <TextInput
                        style={styles.input}
                        placeholder="e.g. How do I configure CORS in FastAPI?"
                        placeholderTextColor="#9ca3af"
                        value={query}
                        onChangeText={setQuery}
                        multiline
                        returnKeyType="search"
                        onSubmitEditing={handleSearch}
                        accessibilityLabel="Query input"
                    />
                </View>

                <TouchableOpacity
                    style={styles.button}
                    onPress={handleSearch}
                    disabled={loading}
                    accessibilityLabel="Search button"
                >
                    <Text style={styles.buttonText}>{loading ? "Searching..." : "🔍 Search"}</Text>
                </TouchableOpacity>

                {/* Loading */}
                {loading && (
                    <ActivityIndicator size="large" color="#6366f1" style={{ marginTop: 24 }} />
                )}

                {/* Error state */}
                {error && (
                    <View style={styles.errorBox}>
                        <Text style={styles.errorText}>⚠️ {error}</Text>
                    </View>
                )}

                {/* Results */}
                {result && (
                    <View style={styles.resultContainer}>
                        {/* Summary Header */}
                        <View style={styles.summaryHeader}>
                            <ConfidenceBadge confidence={result.confidence} />
                            <Text style={styles.metricsText}>
                                {result.latency_ms}ms • {result.tokens_used} tokens
                            </Text>
                        </View>

                        {/* AI Answer Section */}
                        <View style={styles.answerSection}>
                            <View style={styles.sectionHeader}>
                                <Text style={styles.sectionLabel}>AI ANSWER</Text>
                                <Text style={styles.sparkleIcon}>✨</Text>
                            </View>
                            <Text style={styles.answerText}>{result.answer}</Text>
                        </View>

                        {/* Sources Section */}
                        {result.citations.length > 0 && (
                            <View style={styles.sourcesSection}>
                                <Text style={styles.sectionLabel}>SOURCES ({result.citations.length})</Text>
                                <View style={styles.citationsList}>
                                    {result.citations.map((c: Citation, i: number) => (
                                        <CitationCard key={i} citation={c} index={i} />
                                    ))}
                                </View>
                            </View>
                        )}
                    </View>
                )}
            </ScrollView>
        </SafeAreaView>
    );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
    safeArea: { flex: 1, backgroundColor: "#0f172a" },
    container: { padding: 20, paddingBottom: 40 },

    header: { fontSize: 26, fontWeight: "bold", color: "#f8fafc", textAlign: "center", marginTop: 12 },
    subtitle: { fontSize: 14, color: "#94a3b8", textAlign: "center", marginBottom: 24 },

    inputRow: {
        backgroundColor: "#1e293b",
        borderRadius: 12,
        borderWidth: 1,
        borderColor: "#334155",
        marginBottom: 12,
    },
    input: {
        color: "#f1f5f9",
        padding: 14,
        fontSize: 15,
        minHeight: 60,
    },

    button: {
        backgroundColor: "#6366f1",
        borderRadius: 12,
        padding: 16,
        alignItems: "center",
        marginBottom: 24,
    },
    buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },

    errorBox: { backgroundColor: "#450a0a", borderRadius: 8, padding: 12, marginBottom: 16 },
    errorText: { color: "#fca5a5", fontSize: 14 },

    resultContainer: { gap: 16, marginTop: 8 },

    summaryHeader: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 8,
    },
    metricsText: {
        color: "#64748b",
        fontSize: 12,
        fontWeight: "500",
    },

    badge: { borderRadius: 6, paddingHorizontal: 10, paddingVertical: 4 },
    badgeText: { color: "#fff", fontSize: 11, fontWeight: "700", letterSpacing: 0.5 },

    answerSection: {
        backgroundColor: "rgba(99, 102, 241, 0.1)", // Subtle indigo background
        borderRadius: 16,
        padding: 16,
        borderWidth: 1,
        borderColor: "rgba(99, 102, 241, 0.2)",
    },
    sectionHeader: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 8,
    },
    sectionLabel: {
        fontSize: 12,
        fontWeight: "700",
        color: "#6366f1", // Indigo
        letterSpacing: 1.2,
    },
    sparkleIcon: { fontSize: 16 },
    answerText: {
        fontSize: 15,
        color: "#f1f5f9",
        lineHeight: 24,
    },

    sourcesSection: { marginTop: 8 },
    citationsList: { gap: 10 },

    citationCard: {
        backgroundColor: "#1e293b",
        borderRadius: 12,
        padding: 14,
        borderWidth: 1,
        borderColor: "#334155",
    },
    citationHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 6,
    },
    citationNumber: { fontSize: 11, color: "#94a3b8", fontWeight: "700", letterSpacing: 0.5 },
    linkIcon: { fontSize: 14 },
    citationTitle: { fontSize: 14, color: "#f1f5f9", fontWeight: "600", marginBottom: 6 },
    citationPreview: { fontSize: 13, color: "#94a3b8", lineHeight: 20 },
});
