import streamlit as st
import boto3
import json
import os
from botocore.exceptions import NoCredentialsError, NoRegionError, ClientError
import base64
import traceback
import pandas as pd  # Garder seulement si on affiche un DataFrame

st.set_page_config(page_title="RegAI – Analyse réglementaire", layout="centered")
st.title("Analyse d'impact réglementaire via AWS Lambda")
st.caption("Frontend Streamlit minimal – tout le calcul est fait dans Lambda (Bedrock + Kendra)")

# Option: choisir un profil AWS local si vous utilisez aws configure (SSO ou clés)
default_profile = os.getenv("AWS_PROFILE", "")
profile = st.text_input("Profil AWS (optionnel)", value=default_profile, help="Nom du profil configuré via 'aws configure' ou 'aws configure sso'. Laissez vide pour utiliser le profil par défaut.")

# Région et nom de la fonction (surcharge possible)
region = st.text_input("Région AWS", value=os.getenv("AWS_REGION", "us-west-2"))
function_name = st.text_input("Nom de la fonction Lambda", value=os.getenv("REGAI_LAMBDA_NAME", "RegAI-Datathon-Lambda"))

# Option: entrer des clés temporaires (STS) si aucun profil n'est disponible
with st.expander("Authentification alternative (clés temporaires)"):
    use_keys = st.checkbox("Fournir des clés d'accès (non recommandé en production)")
    access_key = secret_key = session_token = ""
    if use_keys:
        access_key = st.text_input("AWS Access Key ID", type="default")
        secret_key = st.text_input("AWS Secret Access Key", type="password")
        session_token = st.text_input("AWS Session Token (optionnel)", type="password")

# Saisie du règlement
uploaded_file = st.file_uploader("Chargez un règlement (txt/htm/html/xml)", type=["txt", "htm", "html", "xml"]) 

# Option debug
show_raw = st.checkbox("Afficher les réponses brutes (debug)", value=False)


def _build_session():
    """Crée et retourne une boto3.Session selon les inputs UI (profil, clés, région)."""
    try:
        if 'use_keys' in globals() and use_keys and access_key and secret_key:
            return boto3.Session(
                aws_access_key_id=access_key.strip(),
                aws_secret_access_key=secret_key.strip(),
                aws_session_token=session_token.strip() or None,
                region_name=region.strip() or "us-west-2",
            )
        if profile.strip():
            return boto3.Session(profile_name=profile.strip(), region_name=region.strip() or None)
        return boto3.Session(region_name=region.strip() or None)
    except Exception:
        raise


def _get_lambda_client():
    sess = _build_session()
    return sess.client("lambda", region_name=region.strip() or "us-west-2")


with st.expander("Diagnostics Lambda"):
    colA, colB = st.columns(2)
    with colA:
        if st.button("DryRun: tester les autorisations (204 attendu)"):
            try:
                lc = _get_lambda_client()
                resp = lc.invoke(
                    FunctionName=function_name.strip() or "RegAI-Datathon-Lambda",
                    InvocationType="DryRun",
                )
                status = resp.get("StatusCode")
                st.success(f"DryRun OK – StatusCode={status} (204 attendu)")
                if show_raw:
                    st.json(resp)
            except ClientError as e:
                err = e.response.get('Error', {})
                st.error(f"ClientError: {err.get('Code')}: {err.get('Message')}")
                if show_raw:
                    st.json(e.response)
            except Exception as e:
                st.error(f"Erreur DryRun: {e}")
                if show_raw:
                    st.code(traceback.format_exc())
    with colB:
        if st.button("Ping Lambda (payload de test) – logs inclus"):
            try:
                lc = _get_lambda_client()
                resp = lc.invoke(
                    FunctionName=function_name.strip() or "RegAI-Datathon-Lambda",
                    InvocationType="RequestResponse",
                    LogType="Tail",
                    Payload=json.dumps({"regulation_text": "ping"}).encode("utf-8"),
                )
                status = resp.get("StatusCode")
                ferror = resp.get("FunctionError")
                req_id = resp.get("ResponseMetadata", {}).get("RequestId")
                st.write(f"StatusCode={status} | FunctionError={ferror or 'None'} | RequestId={req_id}")
                if resp.get("LogResult"):
                    logs = base64.b64decode(resp["LogResult"]).decode("utf-8", errors="ignore")
                    with st.expander("CloudWatch Logs (tail)"):
                        st.code(logs)
                raw_payload = resp.get("Payload")
                final_report = {}
                if raw_payload is not None:
                    outer = json.loads(raw_payload.read().decode("utf-8"))
                    if show_raw:
                        st.subheader("Réponse brute (outer)")
                        st.json(outer)
                    if isinstance(outer, dict):
                        try:
                            final_report = json.loads(outer.get("body", "{}"))
                        except Exception:
                            final_report = outer
                st.subheader("Résultat (parse JSON)")
                st.json(final_report)
            except ClientError as e:
                err = e.response.get('Error', {})
                st.error(f"ClientError: {err.get('Code')}: {err.get('Message')}")
                if show_raw:
                    st.json(e.response)
            except Exception as e:
                st.error(f"Erreur Ping: {e}")
                if show_raw:
                    st.code(traceback.format_exc())

    st.divider()
    if st.button("Afficher la configuration de la fonction (Timeout/Mémoire/VPC)"):
        try:
            lc = _get_lambda_client()
            cfg = lc.get_function_configuration(
                FunctionName=function_name.strip() or "RegAI-Datathon-Lambda"
            )
            timeout = cfg.get("Timeout")
            memory = cfg.get("MemorySize")
            runtime = cfg.get("Runtime")
            last_modified = cfg.get("LastModified")
            role = cfg.get("Role")

            st.write(
                f"Timeout: {timeout}s | Mémoire: {memory} MB | Runtime: {runtime} | Dernière modif: {last_modified}"
            )
            st.write(f"Rôle IAM: {role}")

            vpc_cfg = cfg.get("VpcConfig") or {}
            vpc_id = vpc_cfg.get("VpcId")
            if vpc_id:
                st.warning(
                    "La fonction est rattachée à un VPC (" + vpc_id + "). Assurez-vous d'avoir un NAT Gateway ou des endpoints VPC (Interface) pour les services appelés (ex: Bedrock, Kendra), sinon les appels réseau peuvent expirer."
                )
                with st.expander("Détails VPC"):
                    st.json({
                        "VpcId": vpc_id,
                        "Subnets": vpc_cfg.get("SubnetIds"),
                        "SecurityGroups": vpc_cfg.get("SecurityGroupIds"),
                    })

            if isinstance(timeout, int) and timeout < 30:
                st.error("Timeout actuel très bas. Recommandé: >= 60s pour Bedrock/Kendra. Augmentez à 120s si possible.")

            if show_raw:
                st.subheader("Configuration brute")
                st.json(cfg)
        except ClientError as e:
            err = e.response.get('Error', {})
            st.error(f"ClientError (GetFunctionConfiguration): {err.get('Code')}: {err.get('Message')}")
            if show_raw:
                st.json(e.response)
        except Exception as e:
            st.error(f"Erreur de récupération de la configuration: {e}")
            if show_raw:
                st.code(traceback.format_exc())

if st.button("Lancer l'analyse"):
    if uploaded_file is None:
        st.warning("Veuillez charger un document réglementaire.")
    else:
        regulation_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        with st.spinner("🤖 L'IA analyse l'impact... (cela peut prendre ~30s)"):
            try:
                # 1) Appel de la Lambda (région alignée avec l'infra)
                if use_keys and access_key and secret_key:
                    session = boto3.Session(
                        aws_access_key_id=access_key.strip(),
                        aws_secret_access_key=secret_key.strip(),
                        aws_session_token=session_token.strip() or None,
                        region_name=region.strip() or "us-west-2",
                    )
                elif profile.strip():
                    session = boto3.Session(profile_name=profile.strip(), region_name=region.strip() or None)
                else:
                    session = boto3.Session(region_name=region.strip() or None)
                lambda_client = session.client("lambda", region_name=region.strip() or "us-west-2")
                response = lambda_client.invoke(
                    FunctionName=function_name.strip() or "RegAI-Datathon-Lambda",
                    InvocationType="RequestResponse",
                    LogType="Tail",
                    Payload=json.dumps({"regulation_text": regulation_text}).encode("utf-8"),
                )

                # 2) Récupération du résultat {statusCode, body}
                # Statut et logs
                status = response.get("StatusCode")
                ferror = response.get("FunctionError")
                req_id = response.get("ResponseMetadata", {}).get("RequestId")
                st.write(f"StatusCode={status} | FunctionError={ferror or 'None'} | RequestId={req_id}")
                if response.get("LogResult"):
                    logs = base64.b64decode(response["LogResult"]).decode("utf-8", errors="ignore")
                    with st.expander("CloudWatch Logs (tail)"):
                        st.code(logs)

                raw_payload = response.get("Payload").read()
                outer = json.loads(raw_payload)
                if show_raw:
                    st.subheader("Réponse brute (outer)")
                    st.json(outer)
                final_report = json.loads(outer.get("body", "{}")) if isinstance(outer, dict) else {}

                # 3) Affichage – notre Lambda renvoie { companies: [{company, exposure_reason}], market_data? }
                st.success("Analyse terminée !")

                companies = final_report.get("companies", [])
                if companies:
                    st.subheader("Entreprises les plus exposées")
                    df = pd.DataFrame(companies)
                    # Normaliser colonnes attendues
                    if {"company", "exposure_reason"}.issubset(df.columns):
                        df = df[["company", "exposure_reason"]]
                        df.columns = ["Entreprise", "Raison d'exposition"]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Aucune entreprise détectée dans la réponse.")

                # Market snapshot optionnel si activé côté Lambda
                if "market_data" in final_report:
                    md = final_report["market_data"]
                    st.subheader("Aperçu marché (optionnel)")
                    st.write(f"Titres fusionnés: {md.get('count', 0)}")
                    if md.get("total_market_cap"):
                        st.write(f"Capitalisation totale (approx): {md['total_market_cap']:,}")
                    sample = md.get("sample") or []
                    if sample:
                        st.dataframe(pd.DataFrame(sample))

                # Afficher le JSON brut pour debug
                with st.expander("JSON brut de la Lambda"):
                    st.json(final_report)
            except NoCredentialsError:
                st.error("Identifiants AWS introuvables. Configurez vos identifiants avec 'aws configure' (ou 'aws configure sso') puis relancez. Vous pouvez aussi renseigner le champ 'Profil AWS'.")
                with st.expander("Aide – Configurer les identifiants (Windows PowerShell)"):
                    st.markdown("""
                    1. Installer et connecter l'AWS CLI (si besoin)
                    2. Utiliser un des deux parcours:

                    - Accès par clés (IAM User):
                      - Exécuter:
                        - aws configure
                      - Renseigner AWS Access Key ID, Secret Access Key
                      - Région: us-west-2

                    - Accès SSO (IAM Identity Center):
                      - aws configure sso
                      - Suivre l'authentification navigateur, choisir l'account/role, donner un nom de profil (ex: regai)

                    3. (Optionnel) Sélectionner ce profil dans l'app (champ 'Profil AWS') ou définir la variable d'environnement:
                      - $Env:AWS_PROFILE = "regai"
                    """)
            except (NoRegionError, ClientError) as e:
                st.error(f"Erreur AWS: {e}")
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")
