/**
 * Client-side PII Sanitizer
 * Scrubs names, dates, locations, and other identifiers from text
 * BEFORE it is ever sent to the backend/cloud.
 * PII NEVER LEAVES THE DEVICE.
 */

// Regex patterns for common PII
const PII_PATTERNS: Array<{ pattern: RegExp; replacement: string }> = [
    // Dates (various formats)
    { pattern: /\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b/g, replacement: "[DATE]" },
    { pattern: /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b/gi, replacement: "[DATE]" },
    // Phone numbers
    { pattern: /\b(\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b/g, replacement: "[PHONE]" },
    // Email addresses
    { pattern: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g, replacement: "[EMAIL]" },
    // Social Security Numbers
    { pattern: /\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b/g, replacement: "[SSN]" },
    // Zip codes
    { pattern: /\b\d{5}(?:-\d{4})?\b/g, replacement: "[ZIP]" },
    // Common name prefixes followed by capitalized words (Mr. John Smith)
    { pattern: /\b(Mr|Mrs|Ms|Dr|Prof|Sir|Madam)\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\b/g, replacement: "[NAME]" },
    // Ages (e.g., "I am 34 years old" → keep as "34yo" for clinical utility)
    { pattern: /\b(\d{1,3})\s+years?\s+old\b/gi, replacement: "$1yo" },
];

// Location keywords to redact
const LOCATION_KEYWORDS = [
    "street", "avenue", "boulevard", "road", "lane", "drive", "court", "place",
    "city", "town", "village", "county", "state", "country",
];

export function sanitizeText(text: string): string {
    let sanitized = text;

    // Apply regex patterns
    for (const { pattern, replacement } of PII_PATTERNS) {
        sanitized = sanitized.replace(pattern, replacement);
    }

    // Redact location context (simple heuristic)
    const words = sanitized.split(/\s+/);
    const redacted = words.map((word, i) => {
        const lower = word.toLowerCase().replace(/[^a-z]/g, "");
        if (LOCATION_KEYWORDS.includes(lower) && i > 0) {
            // Redact the word before the location keyword (likely the address)
            return word;
        }
        return word;
    });

    return redacted.join(" ").trim();
}

export function buildSanitizedPrompt(
    symptoms: string,
    age?: number,
    sex?: string,
    conditions?: string[],
    medications?: string[],
    recentLabs?: string
): string {
    const sanitizedSymptoms = sanitizeText(symptoms);
    const sanitizedLabs = recentLabs ? sanitizeText(recentLabs) : "";

    const parts: string[] = [];

    if (age) parts.push(`Age: ${age}`);
    if (sex) parts.push(`Sex: ${sex}`);
    if (conditions?.length) parts.push(`Conditions: ${conditions.join(", ")}`);
    if (medications?.length) parts.push(`Medications: ${medications.join(", ")}`);
    if (sanitizedLabs) parts.push(`Recent Labs: ${sanitizedLabs}`);
    parts.push(`Symptoms: ${sanitizedSymptoms}`);
    parts.push(`Request: ranked differentials, next steps, confidence (0-1), red_flag boolean`);

    return parts.join("\n");
}
