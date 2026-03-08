/**
 * mobile/components/UploadSheet.tsx
 * ────────────────────────────────────
 * Premium slide-up upload modal for the Developer Knowledge Copilot.
 *
 * Features:
 *  - Pick file (30+ formats) with animated progress states
 *  - Paste raw text directly
 *  - Browse all indexed documents
 */

import React, { useState, useRef, useEffect } from "react";
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Modal,
    ScrollView,
    TextInput,
    Animated,
    Dimensions,
    KeyboardAvoidingView,
    Platform,
    ActivityIndicator,
    Alert,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { uploadDocument, listDocuments, uploadText } from "../services/api";

const { height: SCREEN_HEIGHT } = Dimensions.get("window");

// ── Types ──────────────────────────────────────────────────────────────────────

interface IndexedDoc {
    id: number;
    file_name: string;
    chunk_count: number;
    ingested_at: string;
}

type UploadState = "idle" | "uploading" | "success" | "error";
type ActiveTab = "upload" | "paste" | "browse";

interface UploadSheetProps {
    visible: boolean;
    onClose: () => void;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const SUPPORTED_EXTS = [
    ".md", ".txt", ".rst",
    ".pdf", ".docx", ".pptx",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rb", ".php",
    ".java", ".kt", ".scala", ".cs",
    ".go", ".rs", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".swift", ".r", ".lua", ".sh",
];

function getFileIcon(name: string): string {
    const ext = "." + name.split(".").pop()?.toLowerCase();
    if (ext === ".pdf") return "📄";
    if ([".docx", ".doc"].includes(ext)) return "📝";
    if ([".pptx", ".ppt"].includes(ext)) return "📊";
    if ([".py"].includes(ext)) return "🐍";
    if ([".js", ".ts", ".jsx", ".tsx"].includes(ext)) return "🌐";
    if ([".java", ".kt"].includes(ext)) return "☕";
    if ([".go"].includes(ext)) return "🐹";
    if ([".rs"].includes(ext)) return "⚙️";
    if ([".c", ".cpp", ".h"].includes(ext)) return "💻";
    if ([".swift"].includes(ext)) return "🦅";
    if ([".rb"].includes(ext)) return "💎";
    if ([".md", ".txt", ".rst"].includes(ext)) return "📋";
    return "📁";
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function UploadSheet({ visible, onClose }: UploadSheetProps) {
    const [activeTab, setActiveTab] = useState<ActiveTab>("upload");
    const [uploadState, setUploadState] = useState<UploadState>("idle");
    const [uploadedFile, setUploadedFile] = useState<{ name: string; chunks: number } | null>(null);
    const [errorMessage, setErrorMessage] = useState("");

    const [pasteText, setPasteText] = useState("");
    const [pasteTitle, setPasteTitle] = useState("");
    const [pasteState, setPasteState] = useState<UploadState>("idle");

    const [docs, setDocs] = useState<IndexedDoc[]>([]);
    const [loadingDocs, setLoadingDocs] = useState(false);

    const slideAnim = useRef(new Animated.Value(SCREEN_HEIGHT)).current;

    useEffect(() => {
        if (visible) {
            Animated.spring(slideAnim, {
                toValue: 0,
                useNativeDriver: true,
                damping: 20,
                stiffness: 200,
            }).start();
            if (activeTab === "browse") loadDocs();
        } else {
            Animated.timing(slideAnim, {
                toValue: SCREEN_HEIGHT,
                duration: 250,
                useNativeDriver: true,
            }).start();
        }
    }, [visible]);

    useEffect(() => {
        if (activeTab === "browse") loadDocs();
    }, [activeTab]);

    const loadDocs = async () => {
        setLoadingDocs(true);
        try {
            const res = await listDocuments();
            setDocs(res.documents as IndexedDoc[]);
        } catch {
            setDocs([]);
        } finally {
            setLoadingDocs(false);
        }
    };

    const handlePickFile = async () => {
        try {
            const result = await DocumentPicker.getDocumentAsync({
                type: ["*/*"],
                copyToCacheDirectory: true,
            });
            if (result.canceled) return;

            const file = result.assets[0];
            const ext = "." + file.name.split(".").pop()?.toLowerCase();
            if (!SUPPORTED_EXTS.includes(ext)) {
                Alert.alert("Unsupported Type", `Supported: ${SUPPORTED_EXTS.join(", ")}`);
                return;
            }

            setUploadState("uploading");
            setUploadedFile(null);
            setErrorMessage("");

            const res = await uploadDocument(file.uri, file.name, file.mimeType || "text/plain");
            setUploadedFile({ name: res.file_name, chunks: res.chunks_processed });
            setUploadState("success");
        } catch (err: any) {
            setErrorMessage(err.message || "Upload failed");
            setUploadState("error");
        }
    };

    const handlePasteUpload = async () => {
        if (!pasteText.trim()) {
            Alert.alert("Empty", "Please paste some text first.");
            return;
        }
        const title = pasteTitle.trim() || "pasted_document";
        setPasteState("uploading");
        try {
            const res = await uploadText(pasteText, title);
            setPasteState("success");
            setPasteText("");
            setPasteTitle("");
            Alert.alert("✅ Indexed!", `${res.chunks_processed} chunks added from \"${res.file_name}\"`);
        } catch (err: any) {
            setPasteState("error");
            Alert.alert("Upload Failed", err.message);
        } finally {
            setPasteState("idle");
        }
    };

    const handleClose = () => {
        setUploadState("idle");
        setUploadedFile(null);
        setErrorMessage("");
        onClose();
    };

    return (
        <Modal transparent visible={visible} animationType="none" onRequestClose={handleClose}>
            {/* Dim backdrop */}
            <TouchableOpacity style={styles.backdrop} activeOpacity={1} onPress={handleClose} />

            <Animated.View style={[styles.sheet, { transform: [{ translateY: slideAnim }] }]}>
                <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined}>

                    {/* Handle bar */}
                    <View style={styles.handle} />

                    {/* Header */}
                    <View style={styles.sheetHeader}>
                        <Text style={styles.sheetTitle}>📚 Add to Knowledge Base</Text>
                        <TouchableOpacity onPress={handleClose} style={styles.closeBtn}>
                            <Text style={styles.closeBtnText}>✕</Text>
                        </TouchableOpacity>
                    </View>

                    {/* Tab Bar */}
                    <View style={styles.tabBar}>
                        {(["upload", "paste", "browse"] as ActiveTab[]).map((tab) => (
                            <TouchableOpacity
                                key={tab}
                                style={[styles.tab, activeTab === tab && styles.tabActive]}
                                onPress={() => setActiveTab(tab)}
                            >
                                <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>
                                    {tab === "upload" ? "📁 File" : tab === "paste" ? "✏️ Paste" : "📂 Browse"}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </View>

                    {/* ── Tab: File Upload ── */}
                    {activeTab === "upload" && (
                        <View style={styles.tabContent}>
                            <TouchableOpacity
                                style={[
                                    styles.uploadCard,
                                    uploadState === "uploading" && styles.uploadCardLoading,
                                    uploadState === "success" && styles.uploadCardSuccess,
                                    uploadState === "error" && styles.uploadCardError,
                                ]}
                                onPress={uploadState === "idle" || uploadState === "error" ? handlePickFile : undefined}
                                activeOpacity={0.8}
                                disabled={uploadState === "uploading"}
                            >
                                {uploadState === "idle" && (
                                    <>
                                        <Text style={styles.uploadCardIcon}>📄</Text>
                                        <Text style={styles.uploadCardTitle}>Choose a File</Text>
                                        <Text style={styles.uploadCardSub}>
                                            PDF, Word, PowerPoint, Code & more
                                        </Text>
                                        <View style={styles.pillRow}>
                                            {[".pdf", ".docx", ".pptx", ".py", ".java", ".ts"].map((ext) => (
                                                <View key={ext} style={styles.extPill}>
                                                    <Text style={styles.extPillText}>{ext}</Text>
                                                </View>
                                            ))}
                                            <View style={styles.extPill}>
                                                <Text style={styles.extPillText}>+24 more</Text>
                                            </View>
                                        </View>
                                    </>
                                )}
                                {uploadState === "uploading" && (
                                    <>
                                        <ActivityIndicator size="large" color="#6366f1" />
                                        <Text style={styles.uploadCardTitle}>Indexing...</Text>
                                        <Text style={styles.uploadCardSub}>
                                            Chunking, embedding, and storing your document
                                        </Text>
                                    </>
                                )}
                                {uploadState === "success" && (
                                    <>
                                        <Text style={styles.uploadCardIcon}>✅</Text>
                                        <Text style={[styles.uploadCardTitle, { color: "#22c55e" }]}>
                                            Indexed Successfully!
                                        </Text>
                                        <Text style={styles.uploadCardSub}>
                                            {uploadedFile?.name}
                                        </Text>
                                        <View style={styles.chunkBadge}>
                                            <Text style={styles.chunkBadgeText}>
                                                {uploadedFile?.chunks} chunks added to the brain
                                            </Text>
                                        </View>
                                        <TouchableOpacity onPress={() => setUploadState("idle")} style={styles.uploadAgainBtn}>
                                            <Text style={styles.uploadAgainText}>Upload Another</Text>
                                        </TouchableOpacity>
                                    </>
                                )}
                                {uploadState === "error" && (
                                    <>
                                        <Text style={styles.uploadCardIcon}>❌</Text>
                                        <Text style={[styles.uploadCardTitle, { color: "#ef4444" }]}>Upload Failed</Text>
                                        <Text style={styles.uploadCardSub}>{errorMessage}</Text>
                                        <TouchableOpacity onPress={() => setUploadState("idle")} style={styles.uploadAgainBtn}>
                                            <Text style={styles.uploadAgainText}>Try Again</Text>
                                        </TouchableOpacity>
                                    </>
                                )}
                            </TouchableOpacity>
                        </View>
                    )}

                    {/* ── Tab: Paste Text ── */}
                    {activeTab === "paste" && (
                        <View style={styles.tabContent}>
                            <TextInput
                                style={styles.pasteTitle}
                                placeholder="Document name (optional)"
                                placeholderTextColor="#64748b"
                                value={pasteTitle}
                                onChangeText={setPasteTitle}
                            />
                            <TextInput
                                style={styles.pasteInput}
                                placeholder="Paste your text, code, or notes here..."
                                placeholderTextColor="#64748b"
                                value={pasteText}
                                onChangeText={setPasteText}
                                multiline
                                scrollEnabled
                            />
                            <TouchableOpacity
                                style={[styles.pasteBtn, pasteState === "uploading" && { opacity: 0.6 }]}
                                onPress={handlePasteUpload}
                                disabled={pasteState === "uploading"}
                            >
                                {pasteState === "uploading"
                                    ? <ActivityIndicator color="#fff" />
                                    : <Text style={styles.pasteBtnText}>⚡ Index Now</Text>
                                }
                            </TouchableOpacity>
                        </View>
                    )}

                    {/* ── Tab: Browse Docs ── */}
                    {activeTab === "browse" && (
                        <View style={styles.tabContent}>
                            {loadingDocs ? (
                                <ActivityIndicator color="#6366f1" style={{ marginTop: 40 }} />
                            ) : docs.length === 0 ? (
                                <View style={styles.emptyState}>
                                    <Text style={styles.emptyIcon}>📂</Text>
                                    <Text style={styles.emptyText}>No documents indexed yet</Text>
                                    <Text style={styles.emptySub}>Upload your first file to get started</Text>
                                </View>
                            ) : (
                                <ScrollView showsVerticalScrollIndicator={false}>
                                    <Text style={styles.docCount}>{docs.length} document{docs.length !== 1 ? "s" : ""} indexed</Text>
                                    {docs.map((doc) => (
                                        <View key={doc.id} style={styles.docRow}>
                                            <Text style={styles.docIcon}>{getFileIcon(doc.file_name)}</Text>
                                            <View style={styles.docInfo}>
                                                <Text style={styles.docName} numberOfLines={1}>{doc.file_name}</Text>
                                                <Text style={styles.docMeta}>
                                                    {doc.chunk_count} chunks • {new Date(doc.ingested_at).toLocaleDateString()}
                                                </Text>
                                            </View>
                                            <View style={styles.chunkBadgeSmall}>
                                                <Text style={styles.chunkBadgeSmallText}>{doc.chunk_count}</Text>
                                            </View>
                                        </View>
                                    ))}
                                </ScrollView>
                            )}
                        </View>
                    )}
                </KeyboardAvoidingView>
            </Animated.View>
        </Modal>
    );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
    backdrop: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.6)",
    },
    sheet: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: "#0f172a",
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        borderTopWidth: 1,
        borderColor: "rgba(99,102,241,0.2)",
        paddingBottom: 40,
        minHeight: SCREEN_HEIGHT * 0.65,
        maxHeight: SCREEN_HEIGHT * 0.92,
    },
    handle: {
        width: 40, height: 4,
        backgroundColor: "#334155",
        borderRadius: 2,
        alignSelf: "center",
        marginTop: 12,
    },
    sheetHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingHorizontal: 20,
        paddingTop: 16,
        paddingBottom: 12,
    },
    sheetTitle: {
        fontSize: 18,
        fontWeight: "700",
        color: "#f8fafc",
    },
    closeBtn: {
        backgroundColor: "rgba(255,255,255,0.08)",
        borderRadius: 20,
        width: 32, height: 32,
        justifyContent: "center",
        alignItems: "center",
    },
    closeBtnText: { color: "#94a3b8", fontSize: 14 },

    // Tabs
    tabBar: {
        flexDirection: "row",
        marginHorizontal: 16,
        marginBottom: 16,
        backgroundColor: "rgba(255,255,255,0.05)",
        borderRadius: 12,
        padding: 4,
    },
    tab: {
        flex: 1,
        paddingVertical: 8,
        alignItems: "center",
        borderRadius: 10,
    },
    tabActive: {
        backgroundColor: "#6366f1",
    },
    tabText: { fontSize: 12, color: "#64748b", fontWeight: "600" },
    tabTextActive: { color: "#fff" },

    tabContent: {
        paddingHorizontal: 16,
        flex: 1,
    },

    // Upload Card
    uploadCard: {
        backgroundColor: "rgba(99,102,241,0.08)",
        borderWidth: 1.5,
        borderColor: "rgba(99,102,241,0.25)",
        borderRadius: 20,
        borderStyle: "dashed",
        padding: 28,
        alignItems: "center",
        minHeight: 240,
        justifyContent: "center",
    },
    uploadCardLoading: {
        borderColor: "rgba(99,102,241,0.5)",
        borderStyle: "solid",
    },
    uploadCardSuccess: {
        borderColor: "rgba(34,197,94,0.4)",
        backgroundColor: "rgba(34,197,94,0.06)",
        borderStyle: "solid",
    },
    uploadCardError: {
        borderColor: "rgba(239,68,68,0.4)",
        backgroundColor: "rgba(239,68,68,0.06)",
        borderStyle: "solid",
    },
    uploadCardIcon: { fontSize: 48, marginBottom: 12 },
    uploadCardTitle: {
        fontSize: 17,
        fontWeight: "700",
        color: "#f1f5f9",
        marginBottom: 6,
    },
    uploadCardSub: {
        fontSize: 13,
        color: "#64748b",
        textAlign: "center",
        marginBottom: 16,
    },
    pillRow: {
        flexDirection: "row",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: 6,
    },
    extPill: {
        backgroundColor: "rgba(99,102,241,0.15)",
        borderRadius: 8,
        paddingHorizontal: 8,
        paddingVertical: 4,
        margin: 3,
    },
    extPillText: { fontSize: 11, color: "#818cf8" },
    chunkBadge: {
        backgroundColor: "rgba(34,197,94,0.15)",
        borderRadius: 10,
        paddingHorizontal: 14,
        paddingVertical: 6,
        marginBottom: 12,
    },
    chunkBadgeText: { color: "#22c55e", fontSize: 13, fontWeight: "600" },
    uploadAgainBtn: {
        marginTop: 4,
        paddingVertical: 8,
        paddingHorizontal: 20,
        borderRadius: 10,
        backgroundColor: "rgba(99,102,241,0.15)",
    },
    uploadAgainText: { color: "#818cf8", fontWeight: "600", fontSize: 13 },

    // Paste Tab
    pasteTitle: {
        backgroundColor: "rgba(255,255,255,0.05)",
        borderRadius: 10,
        paddingHorizontal: 14,
        paddingVertical: 10,
        color: "#f1f5f9",
        fontSize: 14,
        marginBottom: 10,
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.07)",
    },
    pasteInput: {
        backgroundColor: "rgba(255,255,255,0.05)",
        borderRadius: 14,
        padding: 14,
        color: "#f1f5f9",
        fontSize: 13,
        height: 180,
        textAlignVertical: "top",
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.07)",
        marginBottom: 14,
        fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    },
    pasteBtn: {
        backgroundColor: "#6366f1",
        borderRadius: 14,
        paddingVertical: 14,
        alignItems: "center",
    },
    pasteBtnText: { color: "#fff", fontWeight: "700", fontSize: 15 },

    // Browse Tab
    docCount: {
        color: "#64748b",
        fontSize: 12,
        fontWeight: "600",
        marginBottom: 12,
        textTransform: "uppercase",
        letterSpacing: 0.5,
    },
    docRow: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "rgba(255,255,255,0.04)",
        borderRadius: 14,
        padding: 14,
        marginBottom: 8,
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.06)",
    },
    docIcon: { fontSize: 26, marginRight: 12 },
    docInfo: { flex: 1 },
    docName: { color: "#e2e8f0", fontSize: 14, fontWeight: "600" },
    docMeta: { color: "#475569", fontSize: 12, marginTop: 2 },
    chunkBadgeSmall: {
        backgroundColor: "rgba(99,102,241,0.2)",
        borderRadius: 8,
        paddingHorizontal: 8,
        paddingVertical: 4,
    },
    chunkBadgeSmallText: { color: "#818cf8", fontSize: 12, fontWeight: "700" },

    // Empty State
    emptyState: { flex: 1, alignItems: "center", justifyContent: "center", marginTop: 60 },
    emptyIcon: { fontSize: 48, marginBottom: 12 },
    emptyText: { color: "#94a3b8", fontSize: 17, fontWeight: "600" },
    emptySub: { color: "#475569", fontSize: 13, marginTop: 4 },
});
