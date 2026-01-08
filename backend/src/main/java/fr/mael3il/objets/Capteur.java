package fr.mael3il.objets;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.neo4j.driver.types.Node;

public class Capteur {
    String id;

    @JsonProperty("parcelle_id")
    String parcelleId;

    String type;
    String statut;

    public Capteur(String id, String parcelleId, String type, String statut) {
        this.id = id;
        this.parcelleId = parcelleId;
        this.type = type;
        this.statut = statut;
    }

    public String getId() {
        return id;
    }

    public String getParcelleId() {
        return parcelleId;
    }

    public String getType() {
        return type;
    }

    public String getStatut() {
        return statut;
    }

    public static Capteur from(Node node) {
        return new Capteur(
                node.get("id").asString(),
                node.get("parcelle_id").asString(),
                node.get("type").asString(),
                node.get("statut").asString()
        );
    }
}
