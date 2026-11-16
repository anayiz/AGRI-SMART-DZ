"""Generate the AGRI-SMART DZ competition dossier.

Run from any working directory:
    python tools/generate_gew2026_dossier.py

The generator creates six DOCX reports, seven PDFs (including the deck PDF),
an 8-slide PPTX, and a ZIP archive. PDF export requires LibreOffice on PATH.
CSV inventory and saved model metrics are read from this project. The tool
never deletes or overwrites an existing dossier or archive.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AGRI_SMART_DZ_GEW2026"
ARCHIVE = ROOT / "AGRI_SMART_DZ_GEW2026_Dossier_Complet.zip"
PRICE_DATA = ROOT / "data" / "potato_prices.csv"
STOCK_DATA = ROOT / "data" / "potato_stock.csv"
VIDEO_SCENES = [
    (
        "المشهد 1 — التحدي (0–15 ثانية)",
        "في الجزائر، تؤثر أسعار المنتجات الفلاحية يومياً في الأسر والفلاحين. ومراقبة سعر اليوم وحده لا تكفي؛ بل نحتاج إلى رصد مؤشرات الضغط مبكراً.",
        "سوق خضار جزائري، ثم فلاح في حقل بطاطا. أظهر عبارة «نموذج أولي — بيانات محاكاة».",
        "Cinematic documentary footage of an Algerian neighborhood produce market, a shopper choosing potatoes, then a farmer inspecting a potato field at sunrise. Authentic North African setting, natural light, realistic people, no readable text, no logos, 16:9.",
    ),
    (
        "المشهد 2 — الحل (15–35 ثانية)",
        "AGRI-SMART DZ نموذج ويب لدعم القرار، يجمع مؤشرات الأسعار والمخزون في لوحة معلومات ثنائية اللغة، لتسهيل متابعة الاتجاهات والتحقق منها.",
        "انتقال من السوق إلى لقطة حقيقية للوحة المعلومات. أظهر اسم المشروع وحالة النموذج الأولي، ولا توحِ بوجود اتصال مؤسسي.",
        "A polished transition from an Algerian potato market to a modern agricultural analytics dashboard on a laptop. Subtle data visualization, green and gold palette, leave clean space for a real logo and Arabic or French titles added in post, no legible generated text, 16:9.",
    ),
    (
        "المشهد 3 — الذكاء الاصطناعي (35–60 ثانية)",
        "يقدّر النموذج الأسعار على أفقَي سبعة أيام وثلاثين يوماً. وترصد خوارزمية Isolation Forest الملاحظات غير المعتادة، بينما تصنّف قواعد منفصلة مستوى اليقظة. كل إشارة تحتاج إلى تحقق بشري؛ فهي ليست دليلاً ولا اتهاماً.",
        "تسجيل حقيقي لشاشات التوقعات والشذوذ والمخاطر. أظهر عبارات «بيانات محاكاة» و«تحتاج إلى تحقق» و«تنبيهات قائمة على قواعد».",
        "Close-up of a professional agri-tech dashboard with a stylized Algeria map, a price trend chart and neutral alert indicators. Analytical, restrained government-presentation aesthetic; no accusations, no fabricated numbers, no readable text, 16:9.",
    ),
    (
        "المشهد 4 — المحاكاة (60–75 ثانية)",
        "يتيح المحاكي استكشاف أثر تغيير افتراضي في المخزون على مخرجات النموذج. إنها تجربة حساسية، وليست قياساً للأثر الحقيقي لضخ المخزون الاستراتيجي.",
        "أظهر مؤشر What-If في التطبيق، مع توضيح «محاكاة توضيحية — غير سببية». أضف لقطة قصيرة للنقل واللوجستيات الفلاحية.",
        "A decision-support workshop reviewing an illustrative stock scenario, intercut with a produce truck arriving at an Algerian wholesale market. Collaborative and evidence-led, realistic documentary style, no readable text, 16:9.",
    ),
    (
        "المشهد 5 — الخطوة القادمة (75–90 ثانية)",
        "خطوتنا القادمة هي تجربة ميدانية ببيانات موثقة وشركاء من القطاع. لنجعل التحليل أداة نافعة للفلاح والمستهلك وصانع القرار. AGRI-SMART DZ، نحو سيادة غذائية ذكية.",
        "حقل عند الغروب، ثم شاشة ختامية تضم الشعار واسم الفريق وبيانات الاتصال بعد استكمالها. وضّح أن التجربة الميدانية مقترحة ولم تبدأ بعد.",
        "Hopeful cinematic wide shot of an Algerian farmer at the edge of a potato field at golden hour, gentle breeze and expansive landscape. Leave clean space for a closing slogan and contact details added in post, no generated text, 16:9.",
    ),
]

TECH_SECTIONS = [
    ("1. Résumé", "AGRI-SMART DZ est un prototype web de suivi et d'analyse du marché de la pomme de terre. Il fonctionne sur des données synthétiques et n'est pas connecté aux flux officiels."),
    ("2. Inventaire des données du projet", "Les chemins, lignes, périodes, régions, valeurs manquantes et libellés de source sont inspectés directement depuis les CSV au moment de la génération du dossier."),
    ("3. Sources et stockage", "Les fichiers potato_prices.csv et potato_stock.csv alimentent le pipeline de démonstration. SQLite stocke les observations importées et les saisies manuelles. Aucun connecteur API ministériel ou de marché n'est livré."),
    ("4. Traitement des données", "Le pipeline déduplique les observations, fusionne prix et stocks, puis interpole des valeurs manquantes par région. L'interpolation est une commodité de démonstration et doit être revue pour les données opérationnelles."),
    ("5. Variables et cible du modèle", "Les variables comprennent les retards et moyennes mobiles de prix, les variations de prix et de stock, le ratio de stock et les variables calendaires saisonnières. La cible est le ratio prix futur/prix courant, aux horizons de 7 et 30 jours."),
    ("6. Prévision", "Deux modèles Random Forest produisent des estimations à 7 et 30 jours. Les métriques sauvegardées sont issues du holdout associé aux bundles locaux; elles ne mesurent pas la performance sur des prix réels."),
    ("7. Anomalies et alertes", "Isolation Forest classe des observations de prix et de stock comme inhabituelles. Les niveaux de risque vert/jaune/rouge sont calculés séparément par des règles; aucun de ces signaux ne prouve une fraude ou une rupture d'approvisionnement."),
    ("8. Simulateur What-If", "Le simulateur ajuste le stock d'une région pour la dernière observation puis recalcule les prévisions. Il s'agit d'une analyse de sensibilité du modèle, sans identification causale de l'effet d'une action de stockage ou d'un déstockage SYRPALAC."),
    ("9. Interface et parcours", "Streamlit et Plotly affichent les indicateurs, graphiques, une carte à coordonnées approximatives, les alertes, les anomalies, le simulateur et le formulaire de saisie manuelle. La carte ne représente pas les limites administratives des wilayas."),
    ("10. Architecture et limites d'exploitation", "Le traitement s'exécute dans l'application Streamlit; SQLite constitue le stockage local du prototype. Le projet n'implémente pas l'authentification multi-utilisateur, les rôles ou une architecture de déploiement institutionnelle."),
    ("11. Sécurité et pilote", "Avant tout déploiement partagé, il faut définir le contrôle d'accès, TLS, journalisation, sauvegardes, gouvernance des données et procédures d'incident. Un pilote doit mesurer la qualité des données et les performances contre un baseline."),
    ("12. Validation et évolution", "Les métriques disponibles sont calculées sur un holdout chronologique synthétique. Un pilote doit mesurer MAE/WAPE sur des données autorisées, surveiller les faux positifs et effectuer une revue humaine avant toute décision opérationnelle."),
]

JURY_SECTIONS = [
    ("Besoin et marché à valider", "Le cas d'usage vise le suivi coordonné des prix et stocks de la pomme de terre. Le projet ne dispose pas d'étude de marché chiffrée, d'estimation TAM/SAM/SOM ni de mesure de la demande; ces éléments doivent être établis par des entretiens et un pilote."),
    ("Proposition de valeur", "Transformer des observations autorisées en indicateurs compréhensibles pour orienter le suivi et la vérification humaine. La valeur économique n'a pas encore été mesurée."),
    ("Utilisateurs visés", "À confirmer avec les partenaires du pilote : services agricoles, services de commerce, marchés de gros, coopératives et gestionnaires de stockage."),
    ("État du produit", "Prototype MVP bilingue, démontrable localement, avec données synthétiques. Pas de déploiement institutionnel, de flux temps réel ni d'application Android native."),
    ("Innovation et différenciation recherchées", "Le prototype réunit suivi, prévisions, signaux atypiques, alertes à règles et scénarios dans une interface locale bilingue. L'originalité et l'avantage comparatif restent à évaluer avec des utilisateurs et des données réelles."),
    ("Repères et alternatives", "FEWS NET et FAO FPMA sont cités comme repères internationaux pour l'alerte alimentaire et le suivi des prix, pas comme concurrents directs ni comme preuve de parité fonctionnelle. Les solutions locales existantes doivent être recensées pendant la découverte utilisateur."),
    ("Modèle économique envisagé", "Pistes à valider : pilote institutionnel, hébergement et support dédiés, ou service B2B/B2G. Aucun tarif, revenu, contrat, financement ou partenariat n'est revendiqué."),
    ("Feuille de route", "1) Pilote limité sur la pomme de terre avec données autorisées; 2) évaluation indépendante avant toute extension à d'autres produits; 3) intégration institutionnelle uniquement après accord de partage de données, sécurité et gouvernance."),
    ("Indicateurs de réussite proposés", "Qualité et fraîcheur des données, MAE/WAPE comparées à une baseline saisonnière, délai de détection, taux de faux positifs, adoption par les utilisateurs et coût d'exploitation."),
    ("Risques", "Données tardives ou biaisées, dérive des modèles, fausses alertes, accès non autorisé et confusion entre simulation et observation réelle."),
    ("Demande au jury", "Faciliter une mise en relation pour définir un pilote, un protocole de partage des données et des critères de succès mesurables."),
]

DATA_SECTIONS = [
    ("Jeu de données", "L'inventaire des deux fichiers CSV est calculé à la génération. Les événements injectés servent à tester le comportement du prototype; ils ne décrivent pas des événements réels."),
    ("Préparation", "Les données sont fusionnées par date et région, dédupliquées, puis les valeurs manquantes sont interpolées par région. En production, toute imputation devra être documentée et signalée."),
    ("Caractéristiques", "Prix retardés à 7 et 30 jours, moyennes mobiles à 7 et 30 jours, variations à 7 jours, ratio de stock et encodage saisonnier du mois et du jour de l'année."),
    ("Modèle de prix", "Random Forest Regressor estime un ratio de prix futur/prix courant pour les horizons 7 et 30 jours. Le code entraîne aussi une régression linéaire et compare à la baseline naïve « prix d'aujourd'hui »."),
    ("Séparation temporelle", "Le test chronologique commence en 2026; les lignes d'entraînement sont filtrées afin que leurs cibles futures ne traversent pas la date de coupure."),
    ("Métriques et interprétation", "Le tableau de résultats reprend les métriques enregistrées dans les bundles de modèles présents : MAE et RMSE en DZD/kg, R² sans unité. Elles sont issues du holdout synthétique; elles ne mesurent pas la précision sur le marché algérien."),
    ("Détection d'anomalies", "Isolation Forest est entraîné sur les caractéristiques de prix et de stock. Il signale des points statistiques atypiques; ce n'est pas un détecteur de fraude ni un rapprochement production-stock-vente."),
    ("Risque et alerte", "La classification régionale vert/jaune/rouge utilise des seuils de variation du prix et du stock, distincts du modèle Isolation Forest."),
    ("Évaluation", "Les scores enregistrés dans les bundles proviennent du holdout indiqué par les artefacts locaux. Ils ne doivent pas être présentés comme une précision sur le marché algérien."),
    ("Incertitude", "Les intervalles affichés sont dérivés des résidus synthétiques du jeu de test; leur couverture doit être recalibrée et vérifiée avec des données réelles."),
    ("Étapes préalables au pilote", "Définir les sources, unités, fréquence, règles de qualité, baseline saisonnier, protocole de validation chronologique et mécanisme de revue humaine des alertes."),
]

QA = [
    ("Qu'est-ce qu'AGRI-SMART DZ ?", "Un prototype web d'aide à la décision pour suivre et analyser les prix et stocks de la pomme de terre."),
    ("À quel stade est le projet ?", "Prototype démontrable; pas encore un service opérationnel connecté aux institutions."),
    ("Les prix affichés sont-ils réels ?", "Non. Les données actuelles sont synthétiques et étiquetées comme telles."),
    ("Pourquoi commencer par la pomme de terre ?", "C'est le cas d'usage choisi pour le MVP; l'extension dépendra de données et de validation."),
    ("Quels sont les horizons de prévision ?", "Sept et trente jours après la dernière date du jeu de données."),
    ("Quel modèle prédit les prix ?", "Un RandomForestRegressor distinct est entraîné pour chaque horizon, avec 200 arbres, une profondeur maximale de 12 et un minimum de 5 échantillons par feuille."),
    ("Les scores prouvent-ils la précision réelle ?", "Non. Ils mesurent seulement le comportement sur un holdout synthétique."),
    ("Quelle baseline est actuellement utilisée ?", "Le code compare déjà le modèle au prix courant et à une régression linéaire. Une baseline saisonnière constitue une amélioration à évaluer, pas un résultat déjà obtenu."),
    ("Comment éviter la fuite temporelle ?", "Le découpage chronologique exclut de l'entraînement les lignes dont la cible future chevauche le test."),
    ("Quelles variables sont utilisées ?", "Retards et moyennes mobiles de prix, variations de prix et stock, ratio de stock et variables calendaires."),
    ("Que fait Isolation Forest ?", "Il signale des observations statistiquement inhabituelles à examiner."),
    ("L'algorithme détecte-t-il la spéculation ?", "Non. Un signal statistique n'établit ni intention ni infraction."),
    ("Les alertes sont-elles émises par Isolation Forest ?", "Non. Le niveau de risque est calculé par des règles; Isolation Forest produit séparément les anomalies."),
    ("Le système détecte-t-il les écarts de chaîne d'approvisionnement ?", "Pas actuellement. Cela exige des données vérifiées de production, stockage et volumes de marché."),
    ("Comment fonctionne le simulateur ?", "Il modifie un stock d'entrée puis recalcule une prédiction; c'est une sensibilité du modèle."),
    ("Prouve-t-il l'effet d'un déstockage SYRPALAC ?", "Non. Il ne dispose ni des données opérationnelles ni d'un modèle causal."),
    ("Quelles régions sont représentées ?", "Huit régions simulées; la carte montre des coordonnées de villes approximatives."),
    ("La carte montre-t-elle les limites des wilayas ?", "Non. Elle n'est pas une carte SIG administrative."),
    ("L'application est-elle bilingue ?", "Oui, arabe et français dans l'interface actuelle."),
    ("Existe-t-il un APK ?", "Aucun APK n'est fourni; le produit actuel est une application Web Streamlit."),
    ("Les saisies sont-elles conservées ?", "Les observations manuelles sont stockées dans la base SQLite locale."),
    ("Les alertes sont-elles historisées ?", "Pas dans le parcours actuel; la table existe mais le tableau de bord calcule les alertes à la volée."),
    ("Le produit est-il sécurisé pour plusieurs utilisateurs ?", "Pas encore; authentification, rôles et sécurisation réseau sont préalables à un déploiement partagé."),
    ("Où les données doivent-elles être hébergées ?", "À définir avec les partenaires selon les exigences légales et de gouvernance."),
    ("Comment obtenir des données réelles ?", "Par des accords formels avec les détenteurs et un protocole de qualité et d'usage."),
    ("Quelles données manquent ?", "Prix observés, stocks vérifiés, production, flux d'entrepôt et volumes de marché; météo peut être ajoutée si fiable."),
    ("Comment mesurer l'utilité du pilote ?", "Mesurer la qualité des données, MAE/WAPE face à une baseline, délai de signalement et précision des alertes après revue."),
    ("Quelle est la prochaine étape ?", "Un pilote limité, avec un partenaire, des données autorisées et des critères de succès écrits."),
    ("Le modèle fonctionne-t-il pour d'autres cultures ?", "Ce n'est pas encore validé; chaque filière exige ses propres données et tests."),
    ("Le système réduit-il déjà les prix ?", "Aucun effet économique n'a été mesuré; l'impact est une hypothèse à évaluer."),
    ("Quel est le modèle économique ?", "Des options B2G/B2B sont envisagées, mais elles ne sont pas encore validées commercialement."),
    ("De quoi avez-vous besoin du jury ?", "D'un partenaire pilote, d'un accès encadré aux données et d'experts métier pour la validation."),
]


def inspect_dataset(path: Path, value_column: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required project dataset is missing: {path}")

    required_columns = {"date", "region", "data_source", value_column}
    row_count = 0
    missing_values = 0
    dates = []
    regions = set()
    sources = Counter()
    keys = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise RuntimeError(
                f"Dataset {path.name} is missing columns: {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            row_count += 1
            value_date = (row.get("date") or "").strip()
            region = (row.get("region") or "").strip()
            source = (row.get("data_source") or "").strip() or "UNKNOWN"
            if not value_date or not region:
                raise RuntimeError(f"Dataset {path.name} contains a row without date or region.")
            dates.append(date.fromisoformat(value_date))
            regions.add(region)
            sources[source] += 1
            keys[(value_date, region)] += 1
            if not (row.get(value_column) or "").strip():
                missing_values += 1
    if not row_count:
        raise RuntimeError(f"Dataset {path.name} contains no data rows.")
    return {
        "path": path,
        "rows": row_count,
        "missing_values": missing_values,
        "duplicate_rows": sum(count - 1 for count in keys.values() if count > 1),
        "min_date": min(dates),
        "max_date": max(dates),
        "regions": regions,
        "sources": sources,
    }


def inspect_project_data() -> dict:
    prices = inspect_dataset(PRICE_DATA, "price")
    stock = inspect_dataset(STOCK_DATA, "available_quantity")
    sources = set(prices["sources"]) | set(stock["sources"])
    if sources != {"SYNTHETIC_DEMO"}:
        raise RuntimeError(
            "The dossier currently labels all data as synthetic, but CSV data_source values are "
            + ", ".join(sorted(sources))
            + ". Update and review the dossier content before generation."
        )
    return {"prices": prices, "stock": stock}


def format_data_inventory(facts: dict) -> str:
    prices = facts["prices"]
    stock = facts["stock"]
    regions = prices["regions"] | stock["regions"]
    sources = prices["sources"] + stock["sources"]
    source_summary = ", ".join(f"{name}: {count:,}" for name, count in sorted(sources.items()))
    return (
        f"Inspection directe des CSV du projet : potato_prices.csv contient {prices['rows']:,} lignes "
        f"({prices['missing_values']:,} valeurs de prix vides, {prices['duplicate_rows']:,} doublons date/région); "
        f"potato_stock.csv contient {stock['rows']:,} lignes "
        f"({stock['missing_values']:,} valeurs de stock vides, {stock['duplicate_rows']:,} doublons date/région). "
        f"Les dates observées vont du {min(prices['min_date'], stock['min_date']).isoformat()} au "
        f"{max(prices['max_date'], stock['max_date']).isoformat()}; {len(regions)} libellés de région sont présents. "
        f"Libellés data_source combinés : {source_summary}. Ce relevé décrit les fichiers présents, pas la réalité du marché."
    )


def load_model_metrics() -> tuple[list[list[str]], str]:
    import joblib

    rows = []
    missing_horizons = []
    for horizon in (7, 30):
        model_path = ROOT / "models" / f"price_model_h{horizon}.pkl"
        if not model_path.is_file():
            missing_horizons.append(horizon)
            continue
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or bundle.get("horizon") != horizon:
            raise RuntimeError(f"Model bundle has an invalid horizon or format: {model_path}")
        if bundle.get("trained_on") != "SYNTHETIC_DEMO":
            raise RuntimeError(
                f"Model bundle is not labeled SYNTHETIC_DEMO: {model_path}. Review the dossier claims first."
            )
        metrics = bundle.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(f"Model bundle has no metrics: {model_path}")
        for model_name in ("Naive (price today)", "Linear Regression", "Random Forest"):
            result = metrics.get(model_name)
            if not isinstance(result, dict) or not {"MAE", "RMSE", "R2"}.issubset(result):
                raise RuntimeError(f"Model bundle is missing {model_name} metrics: {model_path}")
            values = [float(result[key]) for key in ("MAE", "RMSE", "R2")]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"Model bundle contains non-finite metrics: {model_path}")
            rows.append(
                [
                    f"{horizon} jours",
                    model_name,
                    f"{values[0]:.3f}",
                    f"{values[1]:.3f}",
                    f"{values[2]:.3f}",
                    str(bundle.get("trained_on", "non renseigné")),
                ]
            )
    if missing_horizons:
        note = (
            "Métriques lues dans les bundles locaux. Bundles absents pour les horizons : "
            + ", ".join(f"{horizon} jours" for horizon in missing_horizons)
            + ". Aucune valeur n'est estimée ou remplacée."
        )
    else:
        note = (
            "Métriques lues dans les bundles locaux; leurs valeurs sont issues du holdout chronologique "
            "enregistré à l'entraînement. Le marqueur trained_on est une métadonnée du bundle, pas une "
            "empreinte cryptographique des CSV ayant servi à l'entraînement."
        )
    return rows, note


def require_dependencies(export_pdf: bool) -> str | None:
    missing = [
        name
        for name in ("docx", "pptx", "joblib", "sklearn")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing document-generation dependencies: "
            + ", ".join(missing)
            + ". Install with: python -m pip install -r tools/requirements-gew2026.txt"
        )
    if export_pdf:
        soffice = shutil.which("soffice")
        if not soffice:
            raise RuntimeError(
                "PDF export was requested, but LibreOffice (soffice) was not found on PATH. "
                "Install LibreOffice or run with --pptx-only."
            )
        return soffice
    return None


def write_docx(
    path: Path,
    title: str,
    subtitle: str,
    sections: list[tuple[str, str]],
    tables: list[tuple[str, list[str], list[list[str]]]] | None = None,
) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(31, 41, 55)

    heading = doc.add_heading(title, 0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading.runs[0].font.color.rgb = RGBColor(27, 67, 50)
    p = doc.add_paragraph(subtitle)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    doc.add_paragraph("AGRI-SMART DZ | Prototype MVP — données simulées")
    doc.add_paragraph("Nom de l'équipe / responsable : [à compléter]")
    for heading_text, body in sections:
        doc.add_heading(heading_text, level=1)
        doc.add_paragraph(body)
    for table_title, headers, rows in tables or []:
        doc.add_heading(table_title, level=1)
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Light Shading Accent 1"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, label in zip(table.rows[0].cells, headers):
            cell.text = label
            for run in cell.paragraphs[0].runs:
                run.bold = True
                run.font.size = Pt(8)
        for row in rows:
            cells = table.add_row().cells
            for cell, value in zip(cells, row):
                cell.text = value
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(8)
    doc.save(path)


def make_video_doc(path: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    doc.add_heading("سيناريو الفيديو — 90 ثانية", 0)
    doc.add_paragraph(
        "خمس مشاهد. تُضاف العناوين والأرقام أثناء المونتاج؛ لا تستخدم نصوصاً مولّدة "
        "بالذكاء الاصطناعي داخل الصور. يجب أن تحمل لقطات التطبيق عبارة «بيانات محاكاة»."
    )
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Shading Accent 1"
    for cell, label in zip(
        table.rows[0].cells,
        ("المشهد / المدة", "التعليق الصوتي", "التوجيه البصري", "English AI video prompt"),
    ):
        cell.text = label
    for scene, voice, visual, prompt in VIDEO_SCENES:
        cells = table.add_row().cells
        for cell, value in zip(cells, (scene, voice, visual, prompt)):
            cell.text = value
    for row in table.rows:
        for column_index, cell in enumerate(row.cells):
            for paragraph in cell.paragraphs:
                if column_index == 1:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    paragraph._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
                for run in paragraph.runs:
                    run.font.size = Pt(8)
                    run.font.color.rgb = RGBColor(31, 41, 55)
                    run.font.name = "Arial"
    doc.save(path)


def make_deck(path: Path) -> None:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    slides = [
        ("AGRI-SMART DZ", ["نحو سيادة غذائية ذكية", "نموذج أولي للبطاطا — جميع بيانات العرض محاكاة", "إعداد: [اسم صاحبة المشروع — يُستكمل]"]),
        ("المشكلة والتحدي", ["صعوبة متابعة الأسعار والمخزون بصورة منسقة.", "الحاجة إلى بيانات سوق موثوقة ومحدّثة.", "التنبيه يوجّه إلى التحقق ولا يوجّه اتهاماً."]),
        ("الحل المقترح", ["لوحة ويب ثنائية اللغة لدعم القرار.", "متابعة الأسعار والمخزون في مناطق العرض.", "توقعات استكشافية لأفق 7 و30 يوماً."]),
        ("الميزات التحليلية", ["Random Forest: تقدير السعر على بيانات اصطناعية.", "Isolation Forest: إبراز أنماط غير معتادة للتحقق.", "مستويات المخاطر محسوبة بقواعد منفصلة عن كشف الشذوذ.", "What-If: تغيير افتراضي للمخزون؛ لا يقيس الأثر السببي لضخ SYRPALAC."]),
        ("البنية التقنية الحالية", ["Python · Streamlit · Scikit-learn · SQLite · Plotly.", "ملفات CSV اصطناعية وإدخال يدوي.", "لا توجد حالياً واجهات API أو مصادقة متعددة المستخدمين."]),
        ("مراجع دولية", ["FEWS NET: متابعة وتحليل مخاطر انعدام الأمن الغذائي الحاد.", "FAO FPMA: متابعة أسعار الأغذية والتنبيه إلى الأسعار المرتفعة.", "التكييف المحلي المقترح يحتاج إلى تحقق عبر تجربة ميدانية."]),
        ("خارطة طريق التوسع", ["1) البطاطا: تجربة ميدانية وقياس MAE/WAPE.", "2) دراسة التوسع للطماطم والبصل والحبوب بعد التحقق.", "3) بحث الربط مع مديريات الفلاحة والتجارة باتفاق رسمي."]),
        ("الأثر والخطوة التالية", ["تحسين وضوح المؤشرات أمام الجهات والفاعلين.", "قياس الأثر قبل ادعاء خفض تقلب الأسعار.", "الخطوة التالية: شريك تجريبي وبيانات موثقة."]),
    ]
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    dark = RGBColor(27, 67, 50)
    gold = RGBColor(212, 163, 115)
    white = RGBColor(255, 255, 255)
    for index, (title, bullets) in enumerate(slides):
        slide = prs.slides.add_slide(blank)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = dark if index == 0 else RGBColor(248, 250, 249)
        title_box = slide.shapes.add_textbox(Inches(0.7), Inches(0.6), Inches(12), Inches(1.0))
        title_paragraph = title_box.text_frame.paragraphs[0]
        title_paragraph.alignment = PP_ALIGN.RIGHT
        title_paragraph._p.get_or_add_pPr().set("rtl", "1")
        title_run = title_paragraph.add_run()
        title_run.text = title
        title_run.font.size = Pt(30)
        title_run.font.bold = True
        title_run.font.color.rgb = gold if index == 0 else dark
        title_run.font.name = "Arial"
        body = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(4.5))
        tf = body.text_frame
        tf.word_wrap = True
        for bullet_index, text in enumerate(bullets):
            paragraph = tf.paragraphs[0] if bullet_index == 0 else tf.add_paragraph()
            paragraph.alignment = PP_ALIGN.RIGHT
            paragraph._p.get_or_add_pPr().set("rtl", "1")
            run = paragraph.add_run()
            run.text = text
            run.font.size = Pt(23 if index == 0 else 20)
            run.font.color.rgb = white if index == 0 else dark
            run.font.name = "Arial"
            paragraph.space_after = Pt(18)
        footer = slide.shapes.add_textbox(Inches(0.7), Inches(7.0), Inches(12), Inches(0.25))
        footer_run = footer.text_frame.paragraphs[0].add_run()
        footer_paragraph = footer.text_frame.paragraphs[0]
        footer_paragraph.alignment = PP_ALIGN.RIGHT
        footer_paragraph._p.get_or_add_pPr().set("rtl", "1")
        footer_run.text = "نموذج أولي • بيانات محاكاة • التنبؤات غير متحققة على بيانات سوق حقيقية"
        footer_run.font.size = Pt(9)
        footer_run.font.color.rgb = gold if index == 0 else dark
        footer_run.font.name = "Arial"
    prs.save(path)


def convert_to_pdf(soffice: str, source: Path) -> Path:
    result = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(source.parent), str(source)],
        capture_output=True,
        text=True,
    )
    pdf_path = source.with_suffix(".pdf")
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"PDF conversion failed for {source.name}: {detail or result.returncode}")
    if not pdf_path.is_file():
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"LibreOffice did not create {pdf_path.name}: {detail or 'no converter output'}")
    return pdf_path


def create_dossier(export_pdf: bool, output_dir: Path | None = None) -> Path:
    soffice = require_dependencies(export_pdf)
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT
    archive_path = ARCHIVE if output_dir == OUTPUT else output_dir.parent / ARCHIVE.name
    if not output_dir.parent.is_dir():
        raise RuntimeError(f"Output parent directory does not exist: {output_dir.parent}")
    if output_dir.exists():
        raise RuntimeError(
            f"Output directory already exists: {output_dir}. Move or rename it before generating; it will not be deleted."
        )
    if archive_path.exists():
        raise RuntimeError(
            f"Archive already exists: {archive_path}. Move or rename it before generating; it will not be overwritten."
        )

    facts = inspect_project_data()
    metric_rows, metric_note = load_model_metrics()
    files = {
        "technical": Path("01_Rapport_Technique/rapport_technique.docx"),
        "jury": Path("02_Project_Report_Jury/rapport_jury.docx"),
        "data_science": Path("03_Rapport_Data_Science_IA/rapport_data_science_ia.docx"),
        "presentation": Path("04_Presentation_Concours/presentation_8_pages.pptx"),
        "video": Path("05_Script_Video/script_video_90s.docx"),
        "demo": Path("06_Guide_Demonstration/guide_demo.docx"),
        "qa": Path("07_Questions_Reponses_Jury/questions_reponses_32.docx"),
    }
    for relative_path in files.values():
        (output_dir / relative_path).parent.mkdir(parents=True, exist_ok=True)

    technical_sections = list(TECH_SECTIONS)
    technical_sections[1] = ("2. Inventaire des données du projet", format_data_inventory(facts))
    data_sections = list(DATA_SECTIONS)
    data_sections[0] = ("Jeu de données — inventaire vérifié", format_data_inventory(facts))
    data_sections.append(("Provenance des métriques", metric_note))

    write_docx(
        output_dir / files["technical"],
        "Rapport technique AGRI-SMART DZ",
        "Architecture, fonctionnement, limites et protocole de pilote",
        technical_sections,
        tables=[
            (
                "Architecture vérifiée dans le code",
                ["Couche", "Fichiers / technologie", "Responsabilité observée"],
                [
                    ["Interface", "app.py · Streamlit", "Sélection de langue et d'instantané; KPI, graphiques, alertes, anomalies, What-If et saisie manuelle."],
                    ["Données", "data/potato_prices.csv · data/potato_stock.csv", "Jeux CSV de démonstration; les libellés de provenance sont inventoriés à l'exécution."],
                    ["Persistance", "src/database.py · agri_smart.db · SQLite", "Import des CSV et stockage local des observations manuelles."],
                    ["Préparation", "src/data_processing.py · Pandas", "Fusion, déduplication, interpolation par région, variables temporelles et découpage chronologique."],
                    ["Prévision", "src/price_prediction.py · Scikit-learn", "Random Forest pour 7 et 30 jours; comparaison à Linear Regression et à la baseline prix courant."],
                    ["Anomalies / risque", "app.py · Isolation Forest et règles", "Signaux atypiques et niveaux de risque sont calculés séparément; aucun ne prouve une infraction."],
                    ["Visualisation", "Plotly", "Courbes, comparaison régionale et points cartographiques à coordonnées de villes approximatives."],
                ],
            )
        ],
    )
    write_docx(
        output_dir / files["jury"],
        "Dossier de présentation au jury",
        "Best Project Competition — GEW 2026 — USTO-MB",
        JURY_SECTIONS,
        tables=[
            (
                "Repères de benchmark — non comparables directement",
                ["Référence", "Périmètre du projet", "Usage comme repère"],
                [
                    ["FEWS NET", "Alerte et analyse de l'insécurité alimentaire", "Inspiration pour la présentation des signaux précoces; pas un concurrent direct."],
                    ["FAO FPMA", "Suivi et analyse des prix alimentaires", "Référence de transparence des prix; pas une preuve de parité fonctionnelle."],
                    ["AGRI-SMART DZ", "Prototype pomme de terre, données de démonstration", "Adaptation locale à valider avec utilisateurs et données autorisées."],
                ],
            )
        ],
    )
    write_docx(
        output_dir / files["data_science"],
        "Rapport Data Science et IA",
        "Méthodes, métriques enregistrées et limites",
        data_sections,
        tables=(
            [
                (
                    "Résultats enregistrés dans les bundles",
                    ["Horizon", "Modèle", "MAE (DZD/kg)", "RMSE (DZD/kg)", "R²", "trained_on"],
                    metric_rows,
                )
            ]
            if metric_rows
            else None
        ),
    )
    make_deck(output_dir / files["presentation"])
    make_video_doc(output_dir / files["video"])
    write_docx(
        output_dir / files["demo"],
        "Guide de démonstration — parcours écran par écran",
        "Parcours conseillé pour une démonstration locale, sans données réelles",
        [
            ("Écran 1 — Lancement et langue", "Depuis la racine du projet, lancer streamlit run app.py. Choisir العربية ou Français dans la barre latérale. Vérifier la mention indiquant que les données sont synthétiques."),
            ("Écran 2 — Instantané et date", "Dans la barre latérale, sélectionner le scénario synthétique de juillet 2026 ou les dernières données disponibles. Lire la date affichée; les prévisions restent calculées à partir de la date la plus récente du jeu de données."),
            ("Écran 3 — Indicateurs et graphiques", "Présenter les quatre indicateurs, puis les courbes de prix et de stock d'Alger, Oran et Constantine. Préciser qu'il s'agit de séries synthétiques; la carte montre des coordonnées de villes approximatives, pas les limites des wilayas."),
            ("Écran 4 — Prévision", "Choisir l'horizon 7 ou 30 jours et une région. Montrer la prévision et son intervalle; rappeler que le holdout et les métriques sont synthétiques et non validés sur le marché."),
            ("Écran 5 — Risque et anomalies", "Distinguer le niveau vert/jaune/rouge, calculé par règles, des points signalés par Isolation Forest. Lire les recommandations comme des demandes de vérification, jamais comme une accusation. Le contrôle production-stock-ventes n'est pas implémenté."),
            ("Écran 6 — Simulateur What-If", "Choisir une région puis faire varier le curseur de stock. Comparer les sorties à la baseline affichée. Dire explicitement que le simulateur modifie une caractéristique d'entrée et ne mesure pas l'effet causal d'un déstockage SYRPALAC."),
            ("Écran 7 — Saisie manuelle", "Saisir uniquement une observation de démonstration clairement identifiée, puis expliquer qu'elle est stockée localement dans SQLite. Ne pas saisir de données personnelles ou opérationnelles sans autorisation."),
            ("Écran 8 — Conclusion", "Résumer ce que le prototype sait faire et ses limites, puis proposer un pilote encadré : données autorisées, partenaire métier, baseline, métriques, revue humaine et critères d'arrêt."),
        ],
    )
    write_docx(
        output_dir / files["qa"],
        "Questions-réponses du jury",
        "32 réponses sur le prototype, les méthodes, les limites et le pilote",
        [(f"Q{i}. {q}", a) for i, (q, a) in enumerate(QA, start=1)],
    )

    docx_sources = [output_dir / relative for relative in files.values() if relative.suffix == ".docx"]
    deck_path = output_dir / files["presentation"]
    pdf_paths = []
    if export_pdf and soffice:
        for source in docx_sources:
            pdf_paths.append(convert_to_pdf(soffice, source))
        pdf_paths.append(convert_to_pdf(soffice, deck_path))

    expected_files = [output_dir / relative for relative in files.values()]
    expected_files.extend(pdf_paths)
    readme = output_dir / "README.txt"
    readme.write_text(
        "AGRI-SMART DZ — dossier GEW 2026\n"
        f"Formats créés : {len(docx_sources)} DOCX, {len(pdf_paths)} PDF, 1 présentation PPTX.\n"
        "Contenu : script vidéo arabe, guide écran par écran et 32 Q&R.\n"
        "L'inventaire des CSV et les métriques sont lus depuis les fichiers du projet au moment de la génération.\n"
        "Les données et résultats affichés restent synthétiques; aucune précision marché, réduction de prix, intégration institutionnelle ou causalité n'est revendiquée.\n"
        "Avec --no-pdf, le générateur produit les DOCX, le PPTX et l'archive sans lancer LibreOffice.\n",
        encoding="utf-8",
    )
    expected_files.append(readme)
    expected_relative = {path.relative_to(output_dir) for path in expected_files}
    actual_relative = {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}
    if actual_relative != expected_relative:
        unexpected = sorted(map(str, actual_relative - expected_relative))
        missing = sorted(map(str, expected_relative - actual_relative))
        raise RuntimeError(f"Output manifest mismatch; missing={missing}; unexpected={unexpected}")

    fd, archive_temp_name = tempfile.mkstemp(prefix=".gew2026-", suffix=".zip", dir=archive_path.parent)
    os.close(fd)
    archive_temp = Path(archive_temp_name)
    try:
        with zipfile.ZipFile(archive_temp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(expected_files):
                zf.write(file, file.relative_to(output_dir.parent))
        with zipfile.ZipFile(archive_temp) as zf:
            bad_member = zf.testzip()
            archived = set(zf.namelist())
        expected_archived = {path.relative_to(output_dir.parent).as_posix() for path in expected_files}
        if bad_member:
            raise RuntimeError(f"Archive integrity check failed at {bad_member}")
        if archived != expected_archived:
            raise RuntimeError("Archive content does not match the verified output manifest.")
        archive_temp.replace(archive_path)
    finally:
        if archive_temp.exists():
            archive_temp.unlink()
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-pdf",
        "--pptx-only",
        dest="export_pdf",
        action="store_false",
        help="Create the DOCX and PPTX files without PDF export or LibreOffice.",
    )
    parser.set_defaults(export_pdf=True)
    args = parser.parse_args()
    try:
        archive = create_dossier(export_pdf=args.export_pdf)
    except (RuntimeError, OSError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Dossier created at: {OUTPUT}")
    print(f"Archive created at: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
