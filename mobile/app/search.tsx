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
import * as DocumentPicker from "expo-document-picker";
import Markdown from "react-native-markdown-display";
import { queryDocuments, uploadDocument, QueryResponse, Citation } from "../services/api";

// ── Confidence badge colors ────────────────────────────────────────────────────
const CONFIDENCE_COLORS: Record<string, string> = {
    high: "#22c55e",
    medium: "#f59e0b",
    low: "#ef4444",
};

// ── Sub-components ────────────────────────────────────────────────────────────

const SearchSkeleton = () => (
    <View style={styles.skeletonContainer}>
        <View style={styles.skeletonHeader}>
            <View style={[styles.skeletonPill, { width: 80 }]} />
            <View style={[styles.skeletonPill, { width: 120 }]} />
        </View>
        <View style={styles.skeletonAnswer}>
            <View style={[styles.skeletonLine, { width: "90%" }]} />
            <View style={[styles.skeletonLine, { width: "95%" }]} />
            <View style={[styles.skeletonLine, { width: "70%" }]} />
        </View>
        <View style={styles.sourcesSection}>
            <View style={[styles.skeletonPill, { width: 100, marginBottom: 12 }]} />
            <View style={styles.citationsList}>
                {[1, 2].map((i) => (
                    <View key={i} style={styles.skeletonCard} />
                ))}
            </View>
        </View>
    </View>
);

const MetricPill = ({ icon, text, color = "#64748b" }: { icon: string, text: string, color?: string }) => (
    <View style={[styles.metricPill, { borderColor: color + "40" }]}>
        <Text style={styles.metricIcon}>{icon}</Text>
        <Text style={[styles.metricText, { color }]}>{text}</Text>
    </View>
);

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

// ── Main Search Screen ────────────────────────────────────────────────────────

export default function SearchScreen() {
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [result, setResult] = useState<QueryResponse | null>(null);
    const [error, setError] = useState<{ message: string, type: "network" | "rate_limit" | "other" } | null>(null);

    const handleUpload = async () => {
        try {
            const result = await DocumentPicker.getDocumentAsync({
                type: ['text/plain', 'text/markdown', '*/*'],
                copyToCacheDirectory: true,
            });

            if (result.canceled) return;

            const file = result.assets[0];

            if (!file.name.endsWith('.md') && !file.name.endsWith('.txt')) {
                Alert.alert("Unsupported File", "Please upload a .txt or .md file.");
                return;
            }

            setUploading(true);
            setError(null);

            const uploadRes = await uploadDocument(file.uri, file.name, file.mimeType || 'text/plain');

            Alert.alert(
                "Upload Complete",
                `Indexed ${uploadRes.chunks_processed} chunks from ${uploadRes.file_name}`
            );

        } catch (err: any) {
            setError({ message: err.message || "Failed to upload document", type: "other" });
        } finally {
            setUploading(false);
        }
    };

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
            let type: "network" | "rate_limit" | "other" = "other";
            let message = err.message ?? "Something went wrong.";

            if (message.includes("429") || message.toLowerCase().includes("rate limit")) {
                type = "rate_limit";
                message = "Whoa, slow down! You've hit the rate limit. Please wait a minute.";
            } else if (message.includes("Aborted") || message.includes("Network request failed")) {
                type = "network";
                message = "Connection lost. Please check your internet and backend status.";
            }

            setError({ message, type });
        } finally {
            setLoading(false);
        }
    };

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">

                {/* Header with Upload Button */}
                <View style={styles.headerRow}>
                    <View>
                        <Text style={styles.header}>🧠 Dev Knowledge Copilot</Text>
                        <Text style={styles.subtitle}>Ask anything about your technical docs</Text>
                    </View>
                    <TouchableOpacity
                        style={styles.uploadButton}
                        onPress={handleUpload}
                        disabled={loading || uploading}
                    >
                        {uploading ? (
                            <ActivityIndicator color="#6366f1" size="small" />
                        ) : (
                            <Text style={styles.uploadIcon}>📁</Text>
                        )}
                    </TouchableOpacity>
                </View>

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

                {/* Skeleton Loader */}
                {loading && <SearchSkeleton />}

                {/* Error state */}
                {error && (
                    <View style={[styles.errorBox, error.type === "rate_limit" && styles.rateLimitBox]}>
                        <Text style={styles.errorText}>
                            {error.type === "network" ? "🌐" : error.type === "rate_limit" ? "⏳" : "⚠️"} {error.message}
                        </Text>
                        {error.type === "network" && (
                            <TouchableOpacity style={styles.retryButton} onPress={handleSearch}>
                                <Text style={styles.retryText}>🔄 Retry Connection</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                )}

                {/* Results */}
                {result && (
                    <View style={styles.resultContainer}>
                        {/* Summary Dashboard */}
                        <View style={styles.metricsDashboard}>
                            <MetricPill
                                icon="⏱️"
                                text={`${result.latency_ms}ms`}
                                color={result.latency_ms < 300 ? "#22c55e" : "#94a3b8"}
                            />
                            <MetricPill
                                icon="🛡️"
                                text={result.confidence.toUpperCase()}
                                color={CONFIDENCE_COLORS[result.confidence]}
                            />
                            <MetricPill
                                icon="💎"
                                text={`${result.tokens_used} tokens`}
                            />
                        </View>

                        {/* AI Answer Section */}
                        <View style={styles.answerSection}>
                            <View style={styles.sectionHeader}>
                                <Text style={styles.sectionLabel}>AI ANSWER</Text>
                                <Text style={styles.sparkleIcon}>✨</Text>
                            </View>

                            <View style={{ flex: 1 }}>
                                <Markdown style={markdownStyles}>
                                    {result.answer}
                                </Markdown>
                            </View>
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
    container: { padding: 20, paddingBottom: 40, flexGrow: 1 },

    headerRow: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "flex-start",
        marginTop: 12,
        marginBottom: 24
    },
    header: { fontSize: 26, fontWeight: "bold", color: "#f8fafc" },
    subtitle: { fontSize: 13, color: "#94a3b8", marginTop: 4 },

    uploadButton: {
        backgroundColor: "rgba(99, 102, 241, 0.15)",
        borderWidth: 1,
        borderColor: "rgba(99, 102, 241, 0.3)",
        borderRadius: 12,
        width: 44,
        height: 44,
        justifyContent: "center",
        alignItems: "center",
    },
    uploadIcon: { fontSize: 20 },

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

    // Skeleton
    skeletonContainer: { gap: 16, marginTop: 8 },
    skeletonHeader: { flexDirection: "row", gap: 8 },
    skeletonPill: { height: 22, backgroundColor: "#1e293b", borderRadius: 6 },
    skeletonAnswer: {
        backgroundColor: "rgba(30, 41, 59, 0.5)",
        borderRadius: 16,
        padding: 16,
        gap: 8,
        borderWidth: 1,
        borderColor: "#1e293b"
    },
    skeletonLine: { height: 12, backgroundColor: "#1e293b", borderRadius: 4 },
    skeletonCard: { height: 100, backgroundColor: "#1e293b", borderRadius: 12 },

    // Metrics Dashboard
    metricsDashboard: {
        flexDirection: "row",
        flexWrap: "wrap",
        gap: 8,
        marginBottom: 8,
    },
    metricPill: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "#1e293b",
        paddingHorizontal: 10,
        paddingVertical: 6,
        borderRadius: 20,
        borderWidth: 1,
    },
    metricIcon: { fontSize: 12, marginRight: 6 },
    metricText: { fontSize: 11, fontWeight: "700", letterSpacing: 0.5 },

    errorBox: { backgroundColor: "#450a0a", borderRadius: 12, padding: 16, marginBottom: 16 },
    rateLimitBox: { backgroundColor: "#1e1b4b", borderColor: "#3730a3", borderWidth: 1 },
    errorText: { color: "#fca5a5", fontSize: 14, lineHeight: 20 },
    retryButton: { marginTop: 12, alignSelf: "flex-start", backgroundColor: "rgba(255,255,255,0.1)", paddingVertical: 8, paddingHorizontal: 12, borderRadius: 8 },
    retryText: { color: "#fff", fontSize: 13, fontWeight: "600" },

    resultContainer: { gap: 16, marginTop: 8 },

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

// ── Markdown Styles ───────────────────────────────────────────────────────────

const markdownStyles: any = {
    body: {
        color: "#f1f5f9",
        fontSize: 15,
        lineHeight: 24,
    },
    strong: {
        fontWeight: "bold",
        color: "#fff",
    },
    em: {
        fontStyle: "italic",
    },
    link: {
        color: "#6366f1",
    },
    bullet_list: {
        marginVertical: 10,
    },
    list_item: {
        marginVertical: 4,
    },
    paragraph: {
        marginTop: 0,
        marginBottom: 10,
    },
    code_inline: {
        backgroundColor: "#2d3748",
        color: "#fbbf24", // Amber color for inline code
        fontFamily: "monospace",
        paddingHorizontal: 4,
        borderRadius: 4,
    },
    code_block: {
        backgroundColor: "#1e293b",
        borderColor: "#334155",
        borderWidth: 1,
        borderRadius: 8,
        padding: 12,
        marginVertical: 10,
        fontFamily: "monospace",
        color: "#94a3b8",
    },
    fence: {
        backgroundColor: "#1e293b",
        borderColor: "#475569",
        borderWidth: 1,
        borderRadius: 8,
        padding: 12,
        marginVertical: 10,
        fontFamily: "monospace",
        color: "#f1f5f9",
    },
};
