/**
 * Local Storage — IndexedDB wrapper for encrypted medical history.
 * Uses the Web Crypto API (AES-GCM) for client-side encryption.
 * Data NEVER leaves the device unencrypted.
 */

const DB_NAME = "medorby_local";
const DB_VERSION = 1;
const STORE_NAME = "medical_history";

// Derive an AES-GCM key from a user passphrase
async function deriveKey(passphrase: string): Promise<CryptoKey> {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
        "raw",
        enc.encode(passphrase),
        { name: "PBKDF2" },
        false,
        ["deriveKey"]
    );
    return crypto.subtle.deriveKey(
        {
            name: "PBKDF2",
            salt: enc.encode("medorby-salt-v1"),
            iterations: 100000,
            hash: "SHA-256",
        },
        keyMaterial,
        { name: "AES-GCM", length: 256 },
        false,
        ["encrypt", "decrypt"]
    );
}

async function encrypt(key: CryptoKey, data: string): Promise<{ iv: string; ciphertext: string }> {
    const enc = new TextEncoder();
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const encrypted = await crypto.subtle.encrypt(
        { name: "AES-GCM", iv },
        key,
        enc.encode(data)
    );
    return {
        iv: btoa(String.fromCharCode(...iv)),
        ciphertext: btoa(String.fromCharCode(...new Uint8Array(encrypted))),
    };
}

async function decrypt(key: CryptoKey, iv: string, ciphertext: string): Promise<string> {
    const dec = new TextDecoder();
    const ivBytes = Uint8Array.from(atob(iv), (c) => c.charCodeAt(0));
    const ciphertextBytes = Uint8Array.from(atob(ciphertext), (c) => c.charCodeAt(0));
    const decrypted = await crypto.subtle.decrypt(
        { name: "AES-GCM", iv: ivBytes },
        key,
        ciphertextBytes
    );
    return dec.decode(decrypted);
}

function openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onupgradeneeded = (e) => {
            const db = (e.target as IDBOpenDBRequest).result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

export interface MedicalEntry {
    id?: number;
    timestamp: number;
    type: "symptom" | "lab" | "medication" | "note";
    content: string; // stored encrypted
}

export async function saveEntry(
    entry: Omit<MedicalEntry, "id">,
    passphrase: string
): Promise<void> {
    const key = await deriveKey(passphrase);
    const { iv, ciphertext } = await encrypt(key, entry.content);
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).add({ ...entry, content: JSON.stringify({ iv, ciphertext }) });
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

export async function getEntries(passphrase: string): Promise<MedicalEntry[]> {
    const key = await deriveKey(passphrase);
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const req = tx.objectStore(STORE_NAME).getAll();
        req.onsuccess = async () => {
            const entries = req.result as MedicalEntry[];
            const decrypted = await Promise.all(
                entries.map(async (e) => {
                    try {
                        const { iv, ciphertext } = JSON.parse(e.content);
                        const content = await decrypt(key, iv, ciphertext);
                        return { ...e, content };
                    } catch {
                        return { ...e, content: "[decryption failed]" };
                    }
                })
            );
            resolve(decrypted);
        };
        req.onerror = () => reject(req.error);
    });
}
