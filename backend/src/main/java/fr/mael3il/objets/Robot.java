package fr.mael3il.objets;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.neo4j.driver.types.Node;

public class Robot {

    String id;
    String modele;
    String type;

    @JsonProperty("weed_detection_threshold")
    Integer weedDetectionThreshold;

    String statut;

    public Robot(String id,
                 String modele,
                 String type,
                 Integer weedDetectionThreshold,
                 String statut) {
        this.id = id;
        this.modele = modele;
        this.type = type;
        this.weedDetectionThreshold = weedDetectionThreshold;
        this.statut = statut;
    }

    public String getId() {
        return id;
    }

    public String getModele() {
        return modele;
    }

    public String getType() {
        return type;
    }

    public Integer getWeedDetectionThreshold() {
        return weedDetectionThreshold;
    }

    public String getStatut() {
        return statut;
    }

    public static Robot from(Node node) {
        return new Robot(
                node.get("id").asString(),
                node.get("modele").asString(),
                node.get("type").asString(),
                node.get("weed_detection_threshold").asInt(),
                node.get("statut").asString()
        );
    }
}
