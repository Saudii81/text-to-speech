/**
 * Language & Dialect Configuration
 * Comprehensive list of languages supported by the Web Speech API
 */

const LANGUAGES = [
    {
        code: 'en-US', name: 'English', flag: '🇺🇸',
        dialects: [
            { code: 'en-US', name: 'United States' },
            { code: 'en-GB', name: 'United Kingdom' },
            { code: 'en-AU', name: 'Australia' },
            { code: 'en-IN', name: 'India' },
            { code: 'en-NZ', name: 'New Zealand' },
            { code: 'en-ZA', name: 'South Africa' },
            { code: 'en-IE', name: 'Ireland' },
            { code: 'en-CA', name: 'Canada' },
        ]
    },
    {
        code: 'es-ES', name: 'Spanish', flag: '🇪🇸',
        dialects: [
            { code: 'es-ES', name: 'Spain' },
            { code: 'es-MX', name: 'Mexico' },
            { code: 'es-AR', name: 'Argentina' },
            { code: 'es-CO', name: 'Colombia' },
            { code: 'es-CL', name: 'Chile' },
            { code: 'es-PE', name: 'Peru' },
        ]
    },
    {
        code: 'fr-FR', name: 'French', flag: '🇫🇷',
        dialects: [
            { code: 'fr-FR', name: 'France' },
            { code: 'fr-CA', name: 'Canada' },
            { code: 'fr-BE', name: 'Belgium' },
            { code: 'fr-CH', name: 'Switzerland' },
        ]
    },
    {
        code: 'de-DE', name: 'German', flag: '🇩🇪',
        dialects: [
            { code: 'de-DE', name: 'Germany' },
            { code: 'de-AT', name: 'Austria' },
            { code: 'de-CH', name: 'Switzerland' },
        ]
    },
    {
        code: 'it-IT', name: 'Italian', flag: '🇮🇹',
        dialects: [
            { code: 'it-IT', name: 'Italy' },
            { code: 'it-CH', name: 'Switzerland' },
        ]
    },
    {
        code: 'pt-BR', name: 'Portuguese', flag: '🇧🇷',
        dialects: [
            { code: 'pt-BR', name: 'Brazil' },
            { code: 'pt-PT', name: 'Portugal' },
        ]
    },
    {
        code: 'zh-CN', name: 'Chinese', flag: '🇨🇳',
        dialects: [
            { code: 'zh-CN', name: 'Mandarin (Simplified)' },
            { code: 'zh-TW', name: 'Mandarin (Traditional)' },
            { code: 'zh-HK', name: 'Cantonese' },
        ]
    },
    {
        code: 'ja-JP', name: 'Japanese', flag: '🇯🇵',
        dialects: [{ code: 'ja-JP', name: 'Japan' }]
    },
    {
        code: 'ko-KR', name: 'Korean', flag: '🇰🇷',
        dialects: [{ code: 'ko-KR', name: 'South Korea' }]
    },
    {
        code: 'ar-SA', name: 'Arabic', flag: '🇸🇦',
        dialects: [
            { code: 'ar-SA', name: 'Saudi Arabia' },
            { code: 'ar-EG', name: 'Egypt' },
            { code: 'ar-AE', name: 'UAE' },
            { code: 'ar-MA', name: 'Morocco' },
            { code: 'ar-DZ', name: 'Algeria' },
        ]
    },
    {
        code: 'hi-IN', name: 'Hindi', flag: '🇮🇳',
        dialects: [{ code: 'hi-IN', name: 'India' }]
    },
    {
        code: 'ru-RU', name: 'Russian', flag: '🇷🇺',
        dialects: [{ code: 'ru-RU', name: 'Russia' }]
    },
    {
        code: 'tr-TR', name: 'Turkish', flag: '🇹🇷',
        dialects: [{ code: 'tr-TR', name: 'Turkey' }]
    },
    {
        code: 'nl-NL', name: 'Dutch', flag: '🇳🇱',
        dialects: [
            { code: 'nl-NL', name: 'Netherlands' },
            { code: 'nl-BE', name: 'Belgium' },
        ]
    },
    {
        code: 'pl-PL', name: 'Polish', flag: '🇵🇱',
        dialects: [{ code: 'pl-PL', name: 'Poland' }]
    },
    {
        code: 'sv-SE', name: 'Swedish', flag: '🇸🇪',
        dialects: [{ code: 'sv-SE', name: 'Sweden' }]
    },
    {
        code: 'th-TH', name: 'Thai', flag: '🇹🇭',
        dialects: [{ code: 'th-TH', name: 'Thailand' }]
    },
    {
        code: 'vi-VN', name: 'Vietnamese', flag: '🇻🇳',
        dialects: [{ code: 'vi-VN', name: 'Vietnam' }]
    },
    {
        code: 'id-ID', name: 'Indonesian', flag: '🇮🇩',
        dialects: [{ code: 'id-ID', name: 'Indonesia' }]
    },
    {
        code: 'ms-MY', name: 'Malay', flag: '🇲🇾',
        dialects: [{ code: 'ms-MY', name: 'Malaysia' }]
    },
    {
        code: 'uk-UA', name: 'Ukrainian', flag: '🇺🇦',
        dialects: [{ code: 'uk-UA', name: 'Ukraine' }]
    },
    {
        code: 'el-GR', name: 'Greek', flag: '🇬🇷',
        dialects: [{ code: 'el-GR', name: 'Greece' }]
    },
    {
        code: 'cs-CZ', name: 'Czech', flag: '🇨🇿',
        dialects: [{ code: 'cs-CZ', name: 'Czech Republic' }]
    },
    {
        code: 'ro-RO', name: 'Romanian', flag: '🇷🇴',
        dialects: [{ code: 'ro-RO', name: 'Romania' }]
    },
    {
        code: 'da-DK', name: 'Danish', flag: '🇩🇰',
        dialects: [{ code: 'da-DK', name: 'Denmark' }]
    },
    {
        code: 'fi-FI', name: 'Finnish', flag: '🇫🇮',
        dialects: [{ code: 'fi-FI', name: 'Finland' }]
    },
    {
        code: 'hu-HU', name: 'Hungarian', flag: '🇭🇺',
        dialects: [{ code: 'hu-HU', name: 'Hungary' }]
    },
    {
        code: 'he-IL', name: 'Hebrew', flag: '🇮🇱',
        dialects: [{ code: 'he-IL', name: 'Israel' }]
    },
    {
        code: 'bn-IN', name: 'Bengali', flag: '🇧🇩',
        dialects: [
            { code: 'bn-IN', name: 'India' },
            { code: 'bn-BD', name: 'Bangladesh' },
        ]
    },
    {
        code: 'ta-IN', name: 'Tamil', flag: '🇮🇳',
        dialects: [{ code: 'ta-IN', name: 'India' }]
    },
];
